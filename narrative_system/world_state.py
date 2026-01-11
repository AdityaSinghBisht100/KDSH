import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Set

@dataclass
class WorldState:
    """
    Entity-Aware World State.
    Instead of a single global state that entangles all character facts,
    we maintain separate BDH states per entity for concentrated signals.
    """
    global_state: torch.Tensor = None           # Shared narrative context
    entity_states: Dict[str, torch.Tensor] = field(default_factory=dict)  # entity_id → BDH state
    known_entities: set = field(default_factory=set)  # All detected entity names
    
    # Timestamps for temporal decay (added in recent update)
    entity_timestamps: Dict[str, float] = field(default_factory=dict)
    global_timestamp: float = 0.0
    
    def get_query_state(self, entity: str, alpha: float = 0.3) -> torch.Tensor:
        """
        Merge global and entity state for inference.
        
        Args:
            entity: Character name to query
            alpha: Weight for global state (0.3 = 30% global, 70% entity)
        
        Returns:
            Combined state tensor
        """
        if entity not in self.entity_states:
            return self.global_state  # Fallback to global
        
        entity_s = self.entity_states[entity]
        if self.global_state is None:
            return entity_s
            
        # Linear interpolation
        return alpha * self.global_state + (1 - alpha) * entity_s


class EntityWriteGate(nn.Module):
    """
    Distinguish the noise from the signal
    Entity Write Gating (Scalar, Dimension-Reduced).
    
    Controls how much new information overwrites entity state.
    prevents entity-state pollution from irrelevant mentions.
    
    FIXED: Rule-based gate (no training required).
    Uses variance difference to detect state-changing chunks.
    """
    def __init__(self, state_dim: int = 64, proj_dim: int = 32):
        super().__init__()
        # No learnable parameters - rule-based
        self.base_gate = 0.1  # Default low gate
        self.sensitivity = 5.0  # How sensitive to variance changes
    
    def forward(self, entity_state: torch.Tensor, chunk_emb: torch.Tensor, 
                global_state: torch.Tensor) -> torch.Tensor:
        """
        Rule-based gate using variance difference.
        
        High gate: chunk brings NEW information (variance increases)
        Low gate: chunk is redundant/descriptive (variance stable)
        """
        # Extract variance features
        e_flat = entity_state.view(-1).float()
        c_flat = chunk_emb.view(-1).float() if chunk_emb is not None else torch.zeros_like(e_flat)
        g_flat = global_state.view(-1).float()
        
        # Compute variance of each state
        e_var = e_flat.var().item() + 1e-8
        c_var = c_flat.var().item() + 1e-8
        g_var = g_flat.var().item() + 1e-8
        
        # Key insight: if chunk variance differs significantly from entity variance,
        # the chunk brings new information → higher gate
        var_ratio = abs(c_var - e_var) / max(e_var, c_var)
        
        # Also consider: if global changed a lot, this is important info
        global_entity_diff = abs(g_var - e_var) / max(e_var, g_var)
        
        # Combine signals
        novelty_score = (var_ratio + global_entity_diff) / 2.0
        
        # Apply sigmoid-like transform with base_gate
        gate = self.base_gate + (1.0 - self.base_gate) * min(novelty_score * self.sensitivity, 1.0)
        
        return torch.tensor(gate, device=entity_state.device)


class AdaptiveMerge(nn.Module):
    """
    Query-Adaptive Global/Entity State Merge.
    
    Learns when to use global vs entity state based on query.
    
    Review Fix #4: Uses distributional summaries (mean + std).
    """
    def __init__(self, state_dim: int, proj_dim: int = 64):
        super().__init__()
        # Uses mean + std = 2x features
        self.alpha_mlp = nn.Sequential(
            nn.Linear(proj_dim * 6, 64),  # 3 inputs * 2 (mean+std)
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.proj_dim = proj_dim
        self.proj = nn.Linear(state_dim, proj_dim) if state_dim != proj_dim else nn.Identity()
    
    def get_summary(self, state: torch.Tensor) -> torch.Tensor:
        """Distributional summary: concat(mean, std)."""
        flat = state.view(-1)
        mean = flat.mean().unsqueeze(0).expand(self.proj_dim // 2)
        std = flat.std().unsqueeze(0).expand(self.proj_dim // 2)
        return torch.cat([mean, std], dim=-1)
    
    def forward(self, statement_emb: torch.Tensor, global_state: torch.Tensor,
                entity_state: torch.Tensor) -> torch.Tensor:
        """
        Returns merged state with learned alpha.
        Bugfix #4: Alpha clamped to [0.05, 0.95] to prevent saturation.
        """
        stmt_summary = self.get_summary(statement_emb) if statement_emb is not None else torch.zeros(self.proj_dim, device=global_state.device)
        global_summary = self.get_summary(global_state)
        entity_summary = self.get_summary(entity_state)
        
        alpha_input = torch.cat([stmt_summary, global_summary, entity_summary], dim=-1)
        alpha = self.alpha_mlp(alpha_input.unsqueeze(0)).squeeze()
        
        # Bugfix #4: Clamp alpha to preserve gradient flow
        alpha = torch.clamp(alpha, 0.05, 0.95)
        
        return alpha * global_state + (1 - alpha) * entity_state
