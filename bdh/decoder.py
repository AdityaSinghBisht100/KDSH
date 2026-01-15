"""
GPT-2 Rationale Decoder Module

Generates human-readable rationales by conditioning GPT-2 on BDH state.
Uses prefix-tuning approach: BDH state is projected to GPT-2 prefix tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .config import BDHConfig


class RationaleDecoder(nn.Module):
    """
    GPT-2 based rationale generator.
    
    Takes the BDH state and generates explanatory text.
    Uses a learned connector to project BDH state into GPT-2's embedding space.
    
    Architecture:
        BDH State [batch, n_neurons]
        -> Connector -> [batch, prefix_len, gpt2_dim]
        -> GPT-2 (frozen) -> Generated text
    """
    
    def __init__(
        self,
        config: BDHConfig,
        freeze_gpt2: bool = True
    ):
        super().__init__()
        self.config = config
        self.prefix_len = config.rationale_prefix_len
        self.gpt2_dim = config.gpt2_dim
        self.n_neurons = config.n_neurons
        
        # Load GPT-2 model and tokenizer
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.gpt2_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.gpt2_model)
        
        # Set pad token (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.gpt2.config.pad_token_id = self.tokenizer.eos_token_id
        
        # Freeze GPT-2 parameters
        if freeze_gpt2:
            for param in self.gpt2.parameters():
                param.requires_grad = False
        
        # Connector: Projects BDH state to GPT-2 prefix
        # BDH state: [batch, n_neurons] -> [batch, prefix_len * gpt2_dim]
        self.connector = nn.Sequential(
            nn.Linear(config.n_neurons, config.n_neurons),
            nn.LayerNorm(config.n_neurons),
            nn.GELU(),
            nn.Linear(config.n_neurons, self.prefix_len * self.gpt2_dim),
            nn.LayerNorm(self.prefix_len * self.gpt2_dim)
        )
        
        # Initialize connector with small weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize connector weights."""
        for module in self.connector.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def get_prefix_embeddings(self, bdh_state: torch.Tensor) -> torch.Tensor:
        """
        Convert BDH state to GPT-2 prefix embeddings.
        
        Args:
            bdh_state: [batch, n_neurons] BDH final state
            
        Returns:
            [batch, prefix_len, gpt2_dim] prefix embeddings
        """
        batch_size = bdh_state.shape[0]
        
        # Project state to prefix space
        prefix_flat = self.connector(bdh_state)  # [batch, prefix_len * gpt2_dim]
        
        # Reshape to [batch, prefix_len, gpt2_dim]
        prefix_embeds = prefix_flat.view(batch_size, self.prefix_len, self.gpt2_dim)
        
        return prefix_embeds
    
    def forward(
        self,
        bdh_state: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training.
        
        Args:
            bdh_state: [batch, n_neurons] BDH final state
            target_ids: [batch, seq_len] target rationale token IDs
            attention_mask: [batch, seq_len] attention mask for targets
            
        Returns:
            (logits, loss) if target_ids provided, else (logits, None)
        """
        batch_size = bdh_state.shape[0]
        device = bdh_state.device
        
        # Get prefix embeddings from BDH state
        prefix_embeds = self.get_prefix_embeddings(bdh_state)  # [batch, prefix_len, gpt2_dim]
        
        if target_ids is not None:
            # Training mode: compute loss
            target_embeds = self.gpt2.transformer.wte(target_ids)  # [batch, seq_len, gpt2_dim]
            
            # Concatenate prefix and target embeddings
            inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)
            # [batch, prefix_len + seq_len, gpt2_dim]
            
            # Create attention mask for full sequence
            prefix_mask = torch.ones(batch_size, self.prefix_len, device=device)
            if attention_mask is not None:
                full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                full_attention_mask = torch.ones(batch_size, inputs_embeds.shape[1], device=device)
            
            # Forward through GPT-2
            outputs = self.gpt2(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            
            logits = outputs.logits  # [batch, prefix_len + seq_len, vocab_size]
            
            # Compute loss only for target tokens (shift by prefix_len)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, self.prefix_len:-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
            
            return logits, loss
        
        else:
            # Inference mode: just return prefix embeddings info
            return prefix_embeds, None
    
    @torch.no_grad()
    def generate(
        self,
        bdh_state: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        prompt: Optional[str] = None
    ) -> List[str]:
        """
        Generate rationale text from BDH state.
        
        Args:
            bdh_state: [batch, n_neurons] BDH final state
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy)
            prompt: Optional prompt to start generation
            
        Returns:
            List of generated rationale strings
        """
        self.eval()
        batch_size = bdh_state.shape[0]
        device = bdh_state.device
        
        # Get prefix embeddings
        prefix_embeds = self.get_prefix_embeddings(bdh_state)  # [batch, prefix_len, gpt2_dim]
        
        # Optional: add a prompt after the prefix
        if prompt is not None:
            prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
            prompt_ids = prompt_ids.expand(batch_size, -1)
            prompt_embeds = self.gpt2.transformer.wte(prompt_ids)
            current_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            generated_length = prompt_ids.shape[1]
        else:
            current_embeds = prefix_embeds
            generated_length = 0
        
        generated_ids = []
        
        # Autoregressive generation
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.gpt2(inputs_embeds=current_embeds, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids.append(next_token)
            
            # Check for EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
            
            # Append next token embedding
            next_embed = self.gpt2.transformer.wte(next_token)  # [batch, 1, gpt2_dim]
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
        
        # Decode generated tokens
        if generated_ids:
            generated_ids = torch.cat(generated_ids, dim=1)  # [batch, generated_len]
            
            # Decode each sequence
            rationales = []
            for i in range(batch_size):
                tokens = generated_ids[i].tolist()
                # Remove EOS tokens
                if self.tokenizer.eos_token_id in tokens:
                    tokens = tokens[:tokens.index(self.tokenizer.eos_token_id)]
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                rationales.append(text.strip())
            
            return rationales
        
        return ["" for _ in range(batch_size)]


class RationaleDecoderWithBDH(nn.Module):
    """
    Convenience wrapper that combines BDH model with RationaleDecoder.
    
    Takes raw SBERT embeddings, processes through BDH, and generates rationale.
    """
    
    def __init__(self, bdh_model: nn.Module, decoder: RationaleDecoder):
        super().__init__()
        self.bdh = bdh_model
        self.decoder = decoder
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        target_rationale_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            inputs_embeds: [batch, seq, sbert_dim] SBERT embeddings
            target_rationale_ids: [batch, rationale_len] target tokens
            
        Returns:
            (logits, loss)
        """
        # Process through BDH
        _, state = self.bdh(inputs_embeds=inputs_embeds)
        
        # Get state representation
        bdh_state = self.bdh.get_state_representation(state)
        
        if bdh_state is None:
            raise ValueError("BDH model did not return a valid state representation")
        
        # Generate rationale
        return self.decoder(bdh_state, target_rationale_ids)
    
    @torch.no_grad()
    def generate_rationale(
        self,
        inputs_embeds: torch.Tensor,
        **generate_kwargs
    ) -> List[str]:
        """
        Generate rationale from SBERT embeddings.
        
        Args:
            inputs_embeds: [batch, seq, sbert_dim] SBERT embeddings
            **generate_kwargs: Arguments passed to decoder.generate()
            
        Returns:
            List of rationale strings
        """
        # Process through BDH
        _, state = self.bdh(inputs_embeds=inputs_embeds)
        
        # Get state representation
        bdh_state = self.bdh.get_state_representation(state)
        
        if bdh_state is None:
            raise ValueError("BDH model did not return a valid state representation")
        
        # Generate
        return self.decoder.generate(bdh_state, **generate_kwargs)
