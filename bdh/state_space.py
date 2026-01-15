"""
BDH State-Space Module

Implements the g-function for gated state updates:
    S_t = (1 - E_t) * α * S_{t-1} + W_t * new_embedding

This allows the model to:
- Forget outdated information (Erase gate)
- Selectively incorporate new information (Write gate)
- Maintain long-term memory via decay α
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BDHStateSpace(nn.Module):
    """
    BDH State-Space with Gated Updates.
    
    Takes semantic embeddings and maintains a recurrent state
    that captures the narrative context.
    """
    
    def __init__(self, embedding_dim: int = 384, n_heads: int = 4, decay: float = 0.95):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        # Learnable decay rate
        self.alpha = nn.Parameter(torch.ones(n_heads, 1) * decay)
        
        # Erase gate: decides what to forget
        self.erase_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
        
        # Write gate: decides how strongly to write
        self.write_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
        
        # State projection
        self.state_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # State buffer: [n_heads, head_dim]
        self.register_buffer("state", torch.zeros(n_heads, self.head_dim))
    
    def reset_state(self):
        """Reset state to zeros."""
        self.state.zero_()
    
    def get_state(self) -> torch.Tensor:
        """Get current state."""
        return self.state.clone()
    
    def set_state(self, state: torch.Tensor):
        """Load a state."""
        self.state.copy_(state.to(self.state.device))
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Update state with new embedding.
        
        Args:
            embedding: [batch, embedding_dim] or [embedding_dim]
            
        Returns:
            Updated state [n_heads, head_dim]
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        # Compute gates
        erase = self.erase_gate(embedding).view(-1, self.n_heads, self.head_dim).mean(0)  # [n_heads, head_dim]
        write = self.write_gate(embedding).view(-1, self.n_heads, self.head_dim).mean(0)
        
        # Project new info
        new_info = self.state_proj(embedding).view(-1, self.n_heads, self.head_dim).mean(0)
        
        # Gated update: S = (1-E) * α * S + W * new
        self.state = (1 - erase) * (self.alpha * self.state) + write * new_info
        
        return self.state
    
    def query(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """
        Query the current state with a query embedding.
        Returns similarity between query and state.
        """
        state_flat = self.state.flatten()  # [embedding_dim]
        query_flat = query_embedding.flatten()
        return F.cosine_similarity(state_flat.unsqueeze(0), query_flat.unsqueeze(0))
