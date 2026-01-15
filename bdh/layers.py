"""
BDH Core Layers

Implements the building blocks from the Dragon Hatchling paper:
- Rotary Position Embedding (RoPE)
- Layer Normalization 
- Positive Sparse Activation (ReLU with sparsity)
- Linear Attention mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RoPE(nn.Module):
    """
    Rotary Position Embedding.
    
    Applies rotation to embeddings based on position, enabling
    the model to understand relative positions without explicit
    position embeddings.
    """
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary embedding.
        
        Args:
            x: [batch, seq_len, dim]
            offset: Position offset for streaming
        
        Returns:
            Rotated tensor of same shape
        """
        seq_len = x.shape[1]
        
        # Extend cache if needed
        if offset + seq_len > self.max_seq_len:
            self._build_cache(offset + seq_len)
        
        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding using the rotation formula."""
        # Split into two halves
        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        
        # Rotate
        rotated = torch.cat((-x2, x1), dim=-1)
        
        # Apply rotation
        return x * cos + rotated * sin


class BDHLayerNorm(nn.Module):
    """
    Layer Normalization for BDH.
    
    Uses RMSNorm variant for stability (no mean subtraction).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class PositiveSparse(nn.Module):
    """
    Positive Sparse Activation.
    
    Implements (...)^+ from the paper: ReLU to ensure x, y ∈ (R+)^n
    This enforces positivity and sparsity in activations.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU (positive part)."""
        return F.relu(x)


class LinearAttention(nn.Module):
    """
    FULLY PARALLEL Linear Attention using vectorized cumulative sum.
    
    Instead of Python for-loops, uses exponential decay weighting
    computed via cumsum for O(seq_len) parallel computation.
    
    Memory: O(batch * seq * dim)
    Speed: Fully vectorized, no Python loops
    """
    def __init__(self, dim: int, decay_rate: float = 0.99):
        super().__init__()
        self.dim = dim
        self.decay_rate = decay_rate
        
        # Learnable decay per dimension
        self.log_decay = nn.Parameter(torch.ones(dim) * math.log(decay_rate))
    
    @property
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.log_decay)
    
    def forward(
        self, 
        x: torch.Tensor,           # [batch, seq, dim]
        v: torch.Tensor,           # [batch, seq, dim]
        state: Optional[torch.Tensor] = None  # [batch, dim] - running state from prev chunk
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fully parallel linear attention using exponential moving average.
        
        Key insight: state_t = decay * state_{t-1} + v_t * x_t
        This is an EMA which can be computed via cumsum with decay weights.
        """
        batch, seq_len, dim = x.shape
        device = x.device
        dtype = x.dtype
        
        # Compute v * x element-wise (the "contribution" at each step)
        contributions = v * x  # [batch, seq, dim]
        
        # Create decay weights: [1, decay, decay^2, ..., decay^{seq-1}]
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        decay_weights = self.decay.view(1, 1, -1) ** positions.view(1, -1, 1)  # [1, seq, dim]
        
        # Scale contributions by inverse decay (so cumsum gives correct result)
        # contribution_t * decay^{-t} -> cumsum -> * decay^t
        inv_decay_weights = 1.0 / (decay_weights + 1e-8)
        scaled_contributions = contributions * inv_decay_weights  # [batch, seq, dim]
        
        # Cumulative sum gives the "un-decayed" running sum
        cumsum = torch.cumsum(scaled_contributions, dim=1)  # [batch, seq, dim]
        
        # Re-apply decay to get proper exponential moving average
        state_seq = cumsum * decay_weights  # [batch, seq, dim]
        
        # Add previous state contribution (decayed appropriately)
        if state is not None:
            # state was the final state from previous chunk
            # It needs to be decayed by positions [1, 2, ..., seq_len]
            state_decay = self.decay.view(1, 1, -1) ** (positions.view(1, -1, 1) + 1)
            state_contribution = state.unsqueeze(1) * state_decay
            state_seq = state_seq + state_contribution
        
        # Compute attention output: a_t = state_t * x_t
        output = state_seq * x  # [batch, seq, dim]
        
        # Final state for next chunk
        final_state = state_seq[:, -1, :]  # [batch, dim]
        
        return output, final_state


class ParallelLinearAttention(nn.Module):
    """
    Alias for LinearAttention (both are now parallel).
    Kept for backward compatibility.
    """
    def __init__(self, dim: int, decay_rate: float = 0.99, chunk_size: int = 256):
        super().__init__()
        self.attention = LinearAttention(dim, decay_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attention(x, v, state)
