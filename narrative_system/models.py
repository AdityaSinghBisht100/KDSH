
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CoherenceClassifier(nn.Module):
    """
    Trained classifier for consistency between Content and Context.
    Input: Content Embedding, Context Embedding
    Output: Logits [Contradict, Consistent]
    """
    def __init__(self, input_dim, device):
        super().__init__()
        self.device = device
        self.use_gnn = False  # GNN removed
        
        # Input: [Isolated_Emb, Contextual_Emb, Diff, Prod]
        # Contextual_Emb is derived from Infinite Memory State
        self.net = nn.Sequential(
            nn.Linear(input_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2) # 0: Contradict, 1: Consistent
        )
        
    def forward(self, iso_emb, ctx_emb):
        """
        Args:
           iso_emb: [B, D] - Embedding of content in isolation
           ctx_emb: [B, D] - Embedding of content given Backstory State
        """
        features = torch.cat([
            iso_emb, 
            ctx_emb, 
            torch.abs(iso_emb - ctx_emb),
            iso_emb * ctx_emb
        ], dim=1)
        return self.net(features)

class NarrativeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
