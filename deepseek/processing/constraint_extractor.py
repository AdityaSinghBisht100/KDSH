import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

class LearnedConstraintDetector(nn.Module):
    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.input_dim = input_dim
        self.constraint_types = [
            "temporal", "causal", "character_trait", "capability", 
            "obligation", "belief", "desire", "prohibition", 
            "event_sequence", "general"
        ]
        
        # Classifier for constraint types
        self.type_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(self.constraint_types))
        )
        
        # Polarity classifier (positive, negative, neutral)
        self.polarity_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # 0: neutral, 1: positive, 2: negative
        )
        
        # Importance scorer (0-1)
        self.importance_scorer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: BDH embeddings [B, D]
        Returns:
            Dictionary with type logits, polarity logits, and importance scores
        """
        type_logits = self.type_classifier(x)
        polarity_logits = self.polarity_classifier(x)
        importance_scores = self.importance_scorer(x)
        
        return {
            "type": type_logits,
            "polarity": polarity_logits,
            "importance": importance_scores
        }

    def extract_constraints_from_narrative(self, embeddings: List[torch.Tensor]) -> List[Dict[str, Any]]:
        self.eval()
        constraints = []
        with torch.no_grad():
            for emb in embeddings:
                outputs = self.forward(emb)
                
                type_idx = torch.argmax(outputs["type"], dim=-1).item()
                polarity_idx = torch.argmax(outputs["polarity"], dim=-1).item()
                importance = outputs["importance"].item()
                
                # Map polarity: 0->0, 1->1, 2->-1
                polarity_map = {0: 0, 1: 1, 2: -1}
                polarity = polarity_map[polarity_idx]
                
                constraints.append({
                    "type": self.constraint_types[type_idx],
                    "polarity": polarity,
                    "importance": importance,
                    "embedding": emb
                })
        return constraints

    def batch_detect(self, x: torch.Tensor) -> List[Dict[str, Any]]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            type_indices = torch.argmax(outputs["type"], dim=-1).cpu().numpy()
            polarity_indices = torch.argmax(outputs["polarity"], dim=-1).cpu().numpy()
            importance_scores = outputs["importance"].cpu().numpy()
            
            results = []
            polarity_map = {0: 0, 1: 1, 2: -1}
            for i in range(len(x)):
                results.append({
                    "type": self.constraint_types[type_indices[i]],
                    "polarity": polarity_map[polarity_indices[i]],
                    "importance": float(importance_scores[i]),
                    "embedding": x[i:i+1]
                })
            return results
