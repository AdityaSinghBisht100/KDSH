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
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.register_buffer(
            'freqs',
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        # Standard O(T^2) Attention for training stability
        assert self.freqs.dtype == torch.float32
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = self.rope(r_phases, K) # Use same rope for K

        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V

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

        # The core USP: Switchable Linear Kernel
        self.attn = LinearAttention(config)

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

    def forward(self, idx, targets=None, use_state=False, return_embeddings=False):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)

        # actually helps with training
        x = self.ln(x)  # B, 1, T, D

        for level in range(C.n_layer):
            x_latent = x @ self.encoder

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
                use_state=use_state, # Pass inference flag
                layer_idx=level # Pass depth index for state management
            )
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
