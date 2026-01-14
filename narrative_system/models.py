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
        
        # Input: [Isolated_Emb, Contextual_Emb, Diff, Prod]
        self.net = nn.Sequential(
            nn.Linear(input_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, iso_emb, ctx_emb):
        features = torch.cat([
            iso_emb, 
            ctx_emb, 
            torch.abs(iso_emb - ctx_emb),
            iso_emb * ctx_emb
        ], dim=1)
        return self.net(features)

class HybridCoherenceClassifier(nn.Module):
    """
    Advanced Hybrid Classifier utilizing Contrastive Energy features.
    """
    def __init__(self, input_dim, device):
        super().__init__()
        self.device = device
        
        # Features: [Iso, Ctx, Neg_Ctx, Iso-Ctx, Iso-NegCtx, Surprise_Ratio]
        # input_dim * 5 + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim * 5 + 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, iso_emb, ctx_emb, neg_ctx_emb, surprise_ratio):
        features = torch.cat([
            iso_emb,
            ctx_emb,
            neg_ctx_emb,
            torch.abs(iso_emb - ctx_emb),
            torch.abs(iso_emb - neg_ctx_emb),
            surprise_ratio.unsqueeze(1)
        ], dim=1)
        return self.net(features)

class NarrativeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
