# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 64 # Increased to 64 (approx 50GB VRAM usage) to fix model collapse
    vocab_size: int = 50257
    # Temporal conditioning parameters
    temporal_dim: int = 64       # Temporal embedding dimension
    use_temporal: bool = True    # Enable temporal-conditioned updates
    max_chapters: int = 100      # Maximum chapter index for embedding


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


# Note: Old O(T²) Attention class removed - using only TemporalLinearAttention now

class LinearAttention(nn.Module):
    """
    BDH State-Space Kernel (Infinite Context USP).
    Implements O(T) recurrent update: S_t = alpha * S_{t-1} + K_t^T V_t
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        head_dim = config.mlp_internal_dim_multiplier * D // nh
        n_layer = config.n_layer
        
        # Learnable decay rates for state persistence (The "Dragon's Long Term Memory")
        self.alpha = nn.Parameter(torch.sigmoid(torch.randn(nh, 1, 1))) 
        
        # Initial State: [Layers, Heads, Key_Dim (N), Value_Dim (D)]
        # Key_Dim = head_dim (from MLP multiplier), Value_Dim = D (from embedding)
        self.register_buffer("state", torch.zeros(n_layer, nh, head_dim, D))
        
    def reset_state(self):
        self.state.zero_()
        
    def forward(self, Q, K, V, use_state=False, layer_idx=0):
        # Q, K, V: [B, nh, T, d]
        B, nh, T, d = Q.size()
        
        if use_state:
            # Recurrent Mode (Inference)
            # This is the "USP" - O(1) step with infinite history in self.state
            out = []
            
            # Select state for this specific layer
            # state_slice is [nh, d, d] (view into the larger buffer)
            # We process sample-by-sample for batching if B>1? 
            # Note: This simple implementation assumes B=1 for correct state update or independent states per batch.
            # For simplicity in this prototype, valid for B=1.
            
            # Access the buffer for this layer. 
            # Note: In-place modification requires care with autograd, but for inference (no_grad) it is fine.
            curr_state = self.state[layer_idx] # [nh, d, d]

            for t in range(T):
                q_t = Q[:, :, t, :].unsqueeze(2) # [B, nh, 1, d]
                k_t = K[:, :, t, :].unsqueeze(2)
                v_t = V[:, :, t, :].unsqueeze(2)
                
                # Update State: S = alpha * S + K^T V
                # We assume B matches state batching. If B > 1, state needs [L, B, H, D, D].
                # Assuming B=1 for Infinite Context generation as per typical use case.
                kv = torch.matmul(k_t.transpose(-1, -2), v_t) # [B, nh, d, d]
                
                # Update globally stored state
                # Note: This update isn't autograd-friendly for BPTT in this specific in-place form.
                # But perfect for Inference.
                if B == 1:
                     curr_state = self.alpha * curr_state + kv.squeeze(0)
                     self.state[layer_idx] = curr_state
                     state_for_attn = curr_state.unsqueeze(0) # [1, nh, d, d]
                else:
                     # Fallback for batching without persistent state logic
                     # (Just behave like local attention or requires expanded state tensor)
                     state_for_attn = torch.matmul(k_t.transpose(-1, -2), v_t) 

                # Attend: O = Q S
                o_t = torch.matmul(q_t, state_for_attn) # [B, nh, 1, d]
                out.append(o_t.squeeze(2))
                
            return torch.stack(out, dim=2)
            
        else:
             # Parallel Mode (Training) - for efficient parallel training on restricted context
             # Here we fall back to standard attention or parallel scan 
             # For this prototype, we stick to standard attention but acknowledge the USP
             return self.standard_attn(Q, K, V)
             
    def standard_attn(self, Q, K, V):
        # Fallback to standard for parallel training speed if linear scan is too slow in python
        scores = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))
        scores = scores.tril(diagonal=0)
        scores = F.softmax(scores, dim=-1)
        return scores @ V
        
    def query_state(self, Q, state, layer_idx=0):
        """
        Read-Only access to the Infinite Context.
        O = Q * S
        """
        # Q: [B, nh, 1, d]
        # state: [L, nh, d, d] or [nh, d, d]
        
        if state.dim() == 4:
            s = state[layer_idx] # [nh, d, d]
        else:
            s = state # [nh, d, d] Assuming passed precise layer state
            
        # O = Q S
        # [B, nh, 1, d] @ [1, nh, d, d] -> [B, nh, 1, d]
        # Need to handle batching if s is not batched.
        
        # Ensure s has batch dim if needed or broadcast
        # S is [nh, d, d]
        # Q is [B, nh, 1, d]
        # We want [B, nh, 1, d] x [B, nh, d, d] -> But S is shared across batch usually?
        # Infinite Context usually implies specific context per Sample.
        # If we have a Batch of queries for the SAME backstory, S is shared.
        # If Batch of queries for DIFFERENT backstories, S must be [B, L, nh, d, d].
        
        # For prototype, assume B=1 or Shared State.
        s = s.unsqueeze(0) # [1, nh, d, d]
        return torch.matmul(Q, s)


class TemporalLinearAttention(LinearAttention):
    """
    Temporal-Conditioned BDH Attention.
    
    Extends LinearAttention with:
    1. Temporal encoding (chapter, scene, timestep)
    2. Erase gate: selectively forget outdated facts
    3. Write gate: control strength of new fact injection
    
    Update rule:
        S_t = (1 - E_t) ⊙ (α_t * S_{t-1}) + W_t ⊙ (K_t^T V_t)
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        D = config.n_embd
        nh = config.n_head
        head_dim = config.mlp_internal_dim_multiplier * D // nh
        τ_dim = config.temporal_dim
        
        # Temporal embeddings
        self.chapter_embed = nn.Embedding(config.max_chapters, τ_dim)
        self.position_scale = nn.Parameter(torch.ones(1) * 0.001)  # Decay factor
        
        # Temporal projections for K, V conditioning
        self.τ_proj_k = nn.Linear(τ_dim, head_dim)
        self.τ_proj_v = nn.Linear(τ_dim, D)
        
        # Gating projections
        # Erase gate: decides what to forget based on query + temporal context
        self.erase_gate = nn.Sequential(
            nn.Linear(head_dim + τ_dim, head_dim),
            nn.Sigmoid()
        )
        # Write gate: decides how strongly to write new facts
        self.write_gate = nn.Sequential(
            nn.Linear(head_dim + τ_dim, head_dim),
            nn.Sigmoid()
        )
        
        # Temporal decay parameter (learnable per-head)
        self.temporal_beta = nn.Parameter(torch.ones(nh, 1, 1) * 0.1)
        
        # Track current timestep for decay computation
        self.register_buffer("current_timestep", torch.tensor(0))
        self.register_buffer("stored_timesteps", torch.zeros(config.n_layer, nh, head_dim, 1))
    
    def encode_temporal(self, chapter_idx: int, timestep: int) -> torch.Tensor:
        """
        Encode temporal position into embedding.
        τ = chapter_embed + positional_encoding(timestep)
        """
        device = self.chapter_embed.weight.device
        
        # Chapter embedding
        chapter_t = torch.tensor([chapter_idx], device=device).clamp(0, self.chapter_embed.num_embeddings - 1)
        τ_chapter = self.chapter_embed(chapter_t)  # [1, τ_dim]
        
        # Positional encoding for fine-grained timestep
        τ_pos = self.position_scale * timestep
        
        return τ_chapter  # [1, τ_dim]
    
    def temporal_decay(self, t_stored: torch.Tensor, t_current: int) -> torch.Tensor:
        """
        Compute temporal decay factor.
        Older facts decay exponentially.
        """
        distance = (t_current - t_stored).abs().float()
        decay = torch.exp(-self.temporal_beta * distance * 0.0001)
        return decay.clamp(0.01, 1.0)  # Never fully forget
    
    def forward(self, Q, K, V, use_state=False, layer_idx=0, return_new_state=False, initial_state=None):
        return self.forward_temporal(Q, K, V, 0, 0, use_state, layer_idx, return_new_state, initial_state)

    def forward_temporal(self, Q, K, V, chapter_idx: int = 0, timestep: int = 0, 
                         use_state: bool = False, layer_idx: int = 0,
                         return_new_state: bool = False, initial_state: torch.Tensor = None):
        """
        Temporal-conditioned forward pass with gated updates.
        """
        B, nh, T, d = Q.size()
        
        # Get temporal encoding
        τ = self.encode_temporal(chapter_idx, timestep)  # [1, τ_dim]
        
        if use_state:
            # Recurrent mode with temporal gating
            out = []
            
            if initial_state is not None:
                curr_state = initial_state
            else:
                curr_state = self.state[layer_idx].clone()  # [nh, head_dim, D]
                
            curr_time = self.stored_timesteps[layer_idx].clone()  # [nh, head_dim, 1]
            
            for t in range(T):
                actual_timestep = timestep + t
                self.current_timestep = torch.tensor(actual_timestep)
                
                q_t = Q[:, :, t, :].unsqueeze(2)  # [B, nh, 1, d]
                k_t = K[:, :, t, :].unsqueeze(2)
                v_t = V[:, :, t, :].unsqueeze(2)
                
                # Temporally condition K, V
                τ_k = self.τ_proj_k(τ).view(1, 1, 1, -1)  # [1, 1, 1, head_dim]
                τ_v = self.τ_proj_v(τ).view(1, 1, 1, -1)  # [1, 1, 1, D]
                k_t = k_t + τ_k
                v_t = v_t + τ_v
                
                # Compute gates
                gate_input = torch.cat([
                    k_t.squeeze(2).mean(dim=0),  # [nh, head_dim]
                    τ.expand(nh, -1)  # [nh, τ_dim]
                ], dim=-1)  # [nh, head_dim + τ_dim]
                
                erase = self.erase_gate(gate_input).unsqueeze(-1)  # [nh, head_dim, 1]
                write = self.write_gate(gate_input).unsqueeze(-1)  # [nh, head_dim, 1]
                
                # Temporal decay on old state
                decay = self.temporal_decay(curr_time, actual_timestep)
                α_t = self.alpha * decay  # [nh, head_dim, 1] ish
                
                # New KV contribution
                kv = torch.matmul(k_t.transpose(-1, -2), v_t).squeeze(0)  # [nh, head_dim, D]
                
                # GATED UPDATE: S = (1-E) * (α * S) + W * KV
                if B == 1:
                    curr_state = (1 - erase) * (α_t * curr_state) + write * kv
                    
                    if not return_new_state:
                        # Inference mode: Persist state
                        with torch.no_grad():
                            self.state[layer_idx] = curr_state.detach().clone()
                            self.stored_timesteps[layer_idx] = torch.full_like(curr_time, actual_timestep)
                            
                    state_for_attn = curr_state.unsqueeze(0)
                else:
                    state_for_attn = kv.unsqueeze(0)
                
                # Attend
                o_t = torch.matmul(q_t, state_for_attn)
                out.append(o_t.squeeze(2))
                
            result = torch.stack(out, dim=2)
            if return_new_state:
                return result, curr_state
            return result
        else:
            # Parallel mode - standard attention
            return self.standard_attn(Q, K, V)


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        # The core USP: Temporal-Conditioned State-Space Attention
        # Always use temporal gating for fact-aware updates
        self.attn = TemporalLinearAttention(config)
        self.use_temporal = True

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def reset_state(self):
        """Reset the internal recurrent state for Infinite Context generation."""
        self.attn.reset_state()
        
    def get_state(self):
        """Return the current recurrent state (The compressed infinite context)."""
        return self.attn.state.clone()
        
    def set_state(self, state):
        """Load a pre-computed recurrent state."""
        self.attn.state.copy_(state)

    def forward(self, idx, targets=None, use_state=False, return_embeddings=False, return_new_state=False):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)

        # actually helps with training
        x = self.ln(x)  # B, 1, T, D
        
        new_states = []

        for level in range(C.n_layer):
            x_latent = x @ self.encoder

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            res = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
                use_state=use_state, # Pass inference flag
                layer_idx=level, # Pass depth index for state management
                return_new_state=return_new_state
            )
            
            if return_new_state:
                yKV, state = res
                new_states.append(state)
            else:
                yKV = res
                
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        if return_embeddings:
            return x.view(B, T, D)
            
        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        if return_new_state:
            return logits, torch.stack(new_states)

        return logits, loss
    
    @torch.no_grad()
    def compute_embeddings(self, idx):
        """Helper to get embeddings (output of last layer before head)"""
        # Call forward asking for embeddings
        emb = self(idx, return_embeddings=True)
        return emb.mean(dim=1) # Mean pooling over T

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        # Standard O(T^2) generation
        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        
    @torch.no_grad()
    def generate_recurrent(
        self,
        idx: torch.Tensor, # Initial prompt
        max_new_tokens: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Infinite Context Generation.
        Maintains internal state across tokens.
        """
        # Prefill state with prompt
        self(idx, use_state=True)
        
        curr_idx = idx[:, -1:]
        all_idx = idx
        
        for _ in range(max_new_tokens):
             logits, _ = self(curr_idx, use_state=True)
             logits = logits[:, -1, :] / temperature
             probs = F.softmax(logits, dim=-1)
             idx_next = torch.multinomial(probs, num_samples=1)
             
             all_idx = torch.cat((all_idx, idx_next), dim=1)
             curr_idx = idx_next
             
        return all_idx
