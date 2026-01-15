"""
BDH-GPU Model

Main model implementation from "The Dragon Hatchling" paper.
Implements Equation 8 and Algorithm in Appendix E.

The core update equations are:
    x_{t,l} = x_{t,l-1} + (D_x · LN(E · y_{t,l-1}))^+
    a*_{t,l} = LinearAttention(x_{t,l}, v*_{t,l-1})
    y_{t,l} = (D_y · LN(a*_{t,l}))^+ ⊙ x_{t,l}
    v*_{t,l} = LN(E · y_{t,l})

Where:
    - x: excitatory activations (positive, sparse)
    - y: output activations (positive, sparse)
    - v*: value vectors for attention
    - a*: attention output
    - D_x, D_y: learnable diagonal scaling matrices
    - E: learnable embedding/projection matrix
    - LN: Layer Normalization
    - (...)^+: ReLU (positive part)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math

from .config import BDHConfig
from .layers import RoPE, BDHLayerNorm, PositiveSparse, LinearAttention


class BDHBlock(nn.Module):
    """
    Single BDH layer block.
    
    Implements one iteration of the BDH update equations.
    """
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        n = config.n_neurons
        d = config.embed_dim
        
        # Layer norm
        self.ln_x = BDHLayerNorm(n)
        self.ln_a = BDHLayerNorm(n)
        self.ln_v = BDHLayerNorm(n)
        
        # Projection matrices
        # E: projects y -> input for x update (and for v* computation)
        self.E = nn.Linear(n, n, bias=False)
        
        # Diagonal scaling matrices (implemented as learnable vectors)
        self.D_x = nn.Parameter(torch.ones(n) * 0.1)
        self.D_y = nn.Parameter(torch.ones(n) * 0.1)
        
        # Positive sparse activation
        self.positive = PositiveSparse()
        
        # Linear attention
        self.attention = LinearAttention(n, decay_rate=config.decay_rate)
        
        # RoPE for position encoding
        self.rope = RoPE(n, base=config.rope_base, max_seq_len=config.max_seq_len)
    
    def forward(
        self,
        x: torch.Tensor,           # [batch, seq, n_neurons]
        y: torch.Tensor,           # [batch, seq, n_neurons]
        state: Optional[Dict[str, torch.Tensor]] = None,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through one BDH block.
        
        Args:
            x: Excitatory activations from previous layer
            y: Output activations from previous layer
            state: Attention state for streaming
            offset: Position offset for RoPE
        
        Returns:
            (new_x, new_y, new_state)
        """
        batch, seq_len, n = x.shape
        
        # Initialize state if needed
        if state is None:
            state = {"sigma": None}
        
        # === Update x ===
        # x_{t,l} = x_{t,l-1} + (D_x · LN(E · y_{t,l-1}))^+
        projected = self.E(y)
        normalized = self.ln_x(projected)
        scaled = self.D_x.view(1, 1, -1) * normalized
        delta_x = self.positive(scaled)
        x_new = x + delta_x
        
        # === Apply RoPE ===
        x_rope = self.rope(x_new, offset=offset)
        
        # === Compute v* ===
        # v*_{t,l} = LN(E · y_{t,l-1})  (using previous y)
        v_star = self.ln_v(self.E(y))
        
        # === Linear Attention ===
        # a*_{t,l} = LinearAttention(x_{t,l}, v*_{t,l-1})
        a_star, new_sigma = self.attention(x_rope, v_star, state.get("sigma"))
        
        # === Update y ===
        # y_{t,l} = (D_y · LN(a*_{t,l}))^+ ⊙ x_{t,l}
        a_normalized = self.ln_a(a_star)
        a_scaled = self.D_y.view(1, 1, -1) * a_normalized
        y_new = self.positive(a_scaled) * x_new
        
        new_state = {"sigma": new_sigma}
        
        return x_new, y_new, new_state


class BDH_GPU(nn.Module):
    """
    Full BDH-GPU Language Model.
    
    Architecture:
        1. Byte embedding: byte -> n-dimensional vector
        2. L layers of BDH blocks
        3. Output projection: n-dimensional -> vocab logits
    
    The model maintains a "synaptic state" σ (sigma) that captures
    the compressed history of all previous tokens. This enables
    infinite context with constant memory.
    """
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        
        # Token embedding (byte-level)
        self.embed = nn.Embedding(config.vocab_size, config.n_neurons)
        
        # BDH layers
        self.layers = nn.ModuleList([
            BDHBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.ln_out = BDHLayerNorm(config.n_neurons)
        self.output = nn.Linear(config.n_neurons, config.vocab_size, bias=False)
        
        # Weight tying (optional, saves parameters)
        # self.output.weight = self.embed.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,   # [batch, seq_len] - byte indices
        state: Optional[List[Dict[str, torch.Tensor]]] = None,
        offset: int = 0
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (bytes 0-255)
            state: List of layer states for streaming
            offset: Position offset for continuing generation
        
        Returns:
            (logits, new_states)
        """
        batch, seq_len = input_ids.shape
        
        # Initialize states
        if state is None:
            state = [None] * self.config.n_layers
        
        # Embed tokens
        x = self.embed(input_ids)  # [batch, seq, n_neurons]
        y = x.clone()  # Initial y = x (identity)
        
        # Pass through BDH layers
        new_states = []
        for i, layer in enumerate(self.layers):
            x, y, layer_state = layer(x, y, state[i], offset)
            new_states.append(layer_state)
        
        # Output projection
        logits = self.output(self.ln_out(y))  # [batch, seq, vocab_size]
        
        return logits, new_states
    
    def get_loss(
        self,
        input_ids: torch.Tensor,
        state: Optional[List[Dict[str, torch.Tensor]]] = None,
        per_sample: bool = False
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Compute language modeling loss.
        
        Args:
            input_ids: [batch, seq_len] token IDs
            state: Optional previous state
            per_sample: If True, return per-sample loss [batch]; if False, return mean scalar
        
        Returns:
            (loss, new_state)
        """
        batch_size, seq_len = input_ids.shape
        
        # Forward pass
        logits, new_state = self.forward(input_ids, state)
        
        # Shift for next-token prediction
        logits_shift = logits[:, :-1, :].contiguous()
        targets_shift = input_ids[:, 1:].contiguous()
        
        if per_sample:
            # Compute loss per sample
            # [batch, seq-1, vocab] -> [batch * (seq-1), vocab]
            logits_flat = logits_shift.view(-1, self.config.vocab_size)
            targets_flat = targets_shift.view(-1)
            
            # Per-token loss
            loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            
            # Reshape to [batch, seq-1] and mean over sequence
            loss_per_token = loss_per_token.view(batch_size, seq_len - 1)
            loss = loss_per_token.mean(dim=1)  # [batch]
        else:
            # Mean loss over all tokens
            loss = F.cross_entropy(
                logits_shift.view(-1, self.config.vocab_size),
                targets_shift.view(-1),
                reduction='mean'
            )
        
        return loss, new_state
    
    def get_perplexity(
        self,
        input_ids: torch.Tensor,
        state: Optional[List[Dict[str, torch.Tensor]]] = None,
        per_sample: bool = False
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Compute perplexity of the input sequence.
        
        Lower perplexity = more expected/consistent.
        Higher perplexity = more surprising/contradictory.
        
        Args:
            per_sample: If True, return per-sample perplexity [batch]
        
        Returns:
            (perplexity, new_state)
        """
        loss, new_state = self.get_loss(input_ids, state, per_sample=per_sample)
        perplexity = torch.exp(loss)
        return perplexity, new_state
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,      # [batch, prompt_len]
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        state: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            prompt: Starting tokens
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            state: Initial state
        
        Returns:
            Generated token IDs [batch, prompt_len + max_new_tokens]
        """
        self.eval()
        tokens = prompt.clone()
        
        for _ in range(max_new_tokens):
            # Get logits for last position only
            logits, state = self.forward(tokens[:, -1:], state, offset=tokens.shape[1] - 1)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=-1)
        
        return tokens
    
    def get_state_representation(
        self,
        state: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Extract a fixed-size representation from the model state.
        
        This can be used for downstream classification tasks.
        
        Args:
            state: List of layer states
        
        Returns:
            [batch, n_neurons] representation vector
        """
        if state is None or len(state) == 0 or state[-1] is None:
            return None
        
        # Use the last layer's sigma (now diagonal: [batch, n])
        sigma = state[-1].get("sigma")
        if sigma is None:
            return None
        
        # sigma is already [batch, n] with diagonal state
        return sigma


class BDHForConsistency(nn.Module):
    """
    BDH model adapted for narrative consistency checking.
    
    Uses the BDH language model as a backbone and adds
    a classification head for consistency prediction.
    """
    def __init__(self, config: BDHConfig, num_classes: int = 2):
        super().__init__()
        self.config = config
        self.bdh = BDH_GPU(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.n_neurons, config.n_neurons // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_neurons // 4, num_classes)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Forward pass for classification.
        
        Returns:
            (class_logits, new_state)
        """
        # Get BDH output
        logits, new_state = self.bdh.forward(input_ids, state)
        
        # Pool: use last token representation
        last_hidden = logits[:, -1, :]  # [batch, vocab_size]
        
        # Actually, we should use the hidden state before output projection
        # Let's modify to extract that
        
        # For now, use state representation
        state_rep = self.bdh.get_state_representation(new_state)
        if state_rep is None:
            state_rep = torch.zeros(input_ids.shape[0], self.config.n_neurons, device=input_ids.device)
        
        # Classify
        class_logits = self.classifier(state_rep)
        
        return class_logits, new_state
