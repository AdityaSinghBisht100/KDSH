import torch
import torch.nn as nn
from typing import Tuple

class ConsistencyClassifier(nn.Module):
    def __init__(self, input_dim: int = 768):
        super().__init__()
        # Cross-attention to weight narrative evidence against backstory
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2) # 0: contradict, 1: consistent
        )

    def forward(self, narrative_emb: torch.Tensor, backstory_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            narrative_emb: [B, S_n, D] sequence of narrative state embeddings
            backstory_emb: [B, S_b, D] sequence of backstory embeddings
        Returns:
            logits: [B, 2]
            weights: [B, S_b, S_n] attention weights
        """
        # Use backstory as query, narrative as key/value
        attn_out, weights = self.attention(backstory_emb, narrative_emb, narrative_emb)
        
        # Pool (mean over sequence)
        pooled_backstory = torch.mean(attn_out, dim=1)
        pooled_narrative = torch.mean(narrative_emb, dim=1)
        
        combined = torch.cat([pooled_backstory, pooled_narrative], dim=-1)
        logits = self.classifier(combined)
        
        return logits, weights