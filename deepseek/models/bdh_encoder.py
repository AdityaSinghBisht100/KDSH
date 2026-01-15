import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[1]
        cos = self.cos_cached[:, offset:offset + seq_len, :]
        sin = self.sin_cached[:, offset:offset + seq_len, :]
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class LinearAttention(nn.Module):
    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Simple causal linear attention: (Q K^T) V is not linear in time
        # The paper uses a specific linear attention. 
        # For now, implementing a basic causal linear attention.
        # state_t = state_{t-1} + V_t K_t^T
        # Out_t = Q_t state_t
        
        # In batch mode:
        # K, V: [B, S, D]
        # We need a causal cumsum.
        
        # Use vectorized implementation for speed
        KV = torch.matmul(K.unsqueeze(-1), V.unsqueeze(-2)) # [B, S, D, D]
        if self.causal:
            states = torch.cumsum(KV, dim=1) # [B, S, D, D]
        else:
            states = torch.sum(KV, dim=1, keepdim=True).expand(-1, Q.size(1), -1, -1)
            
        out = torch.matmul(Q.unsqueeze(-2), states).squeeze(-2) # [B, S, D]
        return out

class BDH_GPU(nn.Module):
    def __init__(self, D=256, H=4, N=32768, L=6, dropout=0.05, vocab_size=256):
        super().__init__()
        self.D = D
        self.H = H
        self.N = N
        self.L = L
        
        self.embed = nn.Embedding(vocab_size, D)
        self.ln_no_affine = nn.LayerNorm(D, elementwise_affine=False)
        
        self.rope = RoPE(D)
        self.attention = LinearAttention(causal=True)
        
        # BDH layer parameters
        self.encoder = nn.Parameter(torch.randn(L, N, D) * 0.02)
        self.decoder_x = nn.Parameter(torch.randn(L, H, D, N // H) * 0.02)
        self.decoder_y = nn.Parameter(torch.randn(L, H, D, N // H) * 0.02)
        self.readout = nn.Parameter(torch.randn(D, vocab_size) * 0.02)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Token embedding -> layer norm
        v_as = self.embed(input_ids)
        v_as = self.ln_no_affine(v_as)
        
        B, S, D = v_as.shape
        
        for l in range(self.L):
            # x = relu(v_as @ decoder_x)
            # decoder_x is [H, D, N//H]. v_as is [B, S, D]
            # Reshape v_as to split into heads or expand decoder?
            # The architecture says decoder_x is [H, D, N//H].
            # This implies multi-head structure.
            
            v_as_heads = v_as.view(B, S, self.H, D // self.H) # Not exactly what paper says
            # Paper: x = relu(v_as @ decoder_x) where decoder_x is [H, D, N//H]
            # This suggests each head processes the WHOLE D-dim vector v_as.
            
            # x_h = relu(v_as @ decoder_x_h) -> [B, S, N//H]
            x_heads = []
            for h in range(self.H):
                x_h = F.relu(torch.matmul(v_as, self.decoder_x[l, h])) # [B, S, N//H]
                x_heads.append(x_h)
            x = torch.cat(x_heads, dim=-1) # [B, S, N]
            
            # attn_out = LinearAttention(Q=x, K=x, V=v_as)
            # Wait, the dimensions of LinearAttention must match.
            # If Q=x, K=x, then x must be D-dim. But x is N-dim.
            # Reread blueprint: "attn_out = LinearAttention(Q=x, K=x, V=v_as)"
            # This implies LinearAttention handles N-dim Q, K and D-dim V.
            # This means state is [N, D].
            
            # Linear Attention with Q=x [B, S, N], K=x [B, S, N], V=v_as [B, S, D]
            # Output: [B, S, D]
            attn_out = self.linear_attention_custom(x, x, v_as) # [B, S, D]
            
            # y = relu(layer_norm(attn_out) @ decoder_y) * x
            # y_h = relu(ln(attn_out) @ decoder_y_h) * x_h
            y_heads = []
            attn_norm = self.ln_no_affine(attn_out)
            for h in range(self.H):
                y_h = F.relu(torch.matmul(attn_norm, self.decoder_y[l, h])) * x_heads[h] # [B, S, N//H]
                y_heads.append(y_h)
            y = torch.cat(y_heads, dim=-1) # [B, S, N]
            
            # y = reshape(y) -> dropout
            y = self.dropout(y)
            
            # v_as = v_as + layer_norm(y @ encoder)
            v_as = v_as + self.ln_no_affine(torch.matmul(y, self.encoder[l]))
            v_as = self.ln_no_affine(v_as)
            
        return torch.matmul(v_as, self.readout)

    def linear_attention_custom(self, Q, K, V):
        # Q: [B, S, N], K: [B, S, N], V: [B, S, D]
        # State: [B, S, N, D]
        # state_t = state_{t-1} + K_t V_t^T (not quite, want state to be N x D)
        # Actually state_t = state_{t-1} + K_t.T @ V_t  where K_t is 1 x N and V_t is 1 x D
        # So state is N x D.
        
        B, S, N = K.shape
        D = V.shape[2]
        
        # Vectorized causal cumsum of (K_t.T @ V_t)
        KV = torch.matmul(K.unsqueeze(-1), V.unsqueeze(-2)) # [B, S, N, D]
        states = torch.cumsum(KV, dim=1) # [B, S, N, D]
        
        # out_t = Q_t @ state_t  where Q_t is 1 x N and state_t is N x D
        out = torch.matmul(Q.unsqueeze(-2), states).squeeze(-2) # [B, S, D]
        return out

class BDHTextEncoder:
    def __init__(self, config):
        self.config = config
        self.model = BDH_GPU(
            D=config.BDH_DIM,
            H=config.BDH_HEADS,
            N=config.BDH_NEURONS,
            L=config.BDH_LAYERS,
            dropout=config.BDH_DROPOUT,
            vocab_size=config.BDH_VOCAB_SIZE
        ).to(config.DEVICE)
        
        self.projection = nn.Linear(config.BDH_DIM, config.HIDDEN_SIZE).to(config.DEVICE)

    def tokenize(self, text: str) -> torch.Tensor:
        # Byte-level tokenization (0-255 vocab)
        tokens = list(text.encode('utf-8', errors='replace'))
        return torch.tensor(tokens, dtype=torch.long, device=self.config.DEVICE).unsqueeze(0)

    def encode_text(self, text: str) -> torch.Tensor:
        input_ids = self.tokenize(text)
        with torch.no_grad():
            # Get v_as from the model. 
            # We need to expose v_as or modify forward.
            # Let's modify BDH_GPU.forward to return (logits, v_as)
            logits, v_as = self.model.forward_with_embeddings(input_ids)
            # Use last token embedding or mean? Blueprint says "Projects BDH embeddings to standard 768D"
            # Usually we take the last state for sequence representation.
            pooled = v_as[:, -1, :] # [B, D]
            projected = self.projection(pooled) # [B, 768]
        return projected

    def encode_long_document(self, text: str, chunk_size: int = 2000) -> List[torch.Tensor]:
        # Process in chunks
        embeddings = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            emb = self.encode_text(chunk)
            embeddings.append(emb)
        return embeddings

# Modifying BDH_GPU to return embeddings
def forward_with_embeddings(self, input_ids: torch.Tensor):
    v_as = self.embed(input_ids)
    v_as = self.ln_no_affine(v_as)
    
    B, S, D = v_as.shape
    
    for l in range(self.L):
        x_heads = []
        for h in range(self.H):
            x_h = F.relu(torch.matmul(v_as, self.decoder_x[l, h]))
            x_heads.append(x_h)
        x = torch.cat(x_heads, dim=-1)
        
        attn_out = self.linear_attention_custom(x, x, v_as)
        
        y_heads = []
        attn_norm = self.ln_no_affine(attn_out)
        for h in range(self.H):
            y_h = F.relu(torch.matmul(attn_norm, self.decoder_y[l, h])) * x_heads[h]
            y_heads.append(y_h)
        y = torch.cat(y_heads, dim=-1)
        
        y = self.dropout(y)
        v_as = v_as + self.ln_no_affine(torch.matmul(y, self.encoder[l]))
        v_as = self.ln_no_affine(v_as)
        
    logits = torch.matmul(v_as, self.readout)
    return logits, v_as

BDH_GPU.forward_with_embeddings = forward_with_embeddings
