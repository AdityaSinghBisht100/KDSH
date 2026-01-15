"""
Consistency Training Pipeline

Trains a simple classifier on top of semantic similarity scores.
"""
import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from typing import Tuple

class ConsistencyClassifier(nn.Module):
    """
    Simple classifier that takes:
    - similarity score between statement and backstory
    - statement embedding
    - backstory embedding
    
    And predicts: consistent (1) or contradict (0)
    """
    def __init__(self, embedding_dim: int = 384):
        super().__init__()
        # Input: [similarity, stmt_emb, backstory_emb] -> 1 + 384 + 384 = 769
        self.classifier = nn.Sequential(
            nn.Linear(1 + embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # 2 classes: consistent, contradict
        )
    
    def forward(self, similarity: torch.Tensor, stmt_emb: torch.Tensor, backstory_emb: torch.Tensor):
        """
        Args:
            similarity: [batch] cosine similarity scores
            stmt_emb: [batch, emb_dim]
            backstory_emb: [batch, emb_dim]
        """
        if similarity.dim() == 0:
            similarity = similarity.unsqueeze(0)
        if similarity.dim() == 1:
            similarity = similarity.unsqueeze(-1)
        
        features = torch.cat([similarity, stmt_emb, backstory_emb], dim=-1)
        return self.classifier(features)


def train_consistency_model(
    embedder,
    memory,
    train_df: pd.DataFrame,
    device: str = "cuda",
    epochs: int = 10,
    lr: float = 1e-3
) -> ConsistencyClassifier:
    """
    Train the consistency classifier.
    """
    print("\n=== Phase 2: Training Consistency Classifier ===")
    
    classifier = ConsistencyClassifier(embedder.embedding_dim).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare data
    samples = []
    for _, row in train_df.iterrows():
        book = str(row.get('book_name', '')).strip()
        char = str(row.get('char', '')).strip()
        content = str(row.get('content', '')).strip()
        label = str(row.get('label', '')).strip().lower()
        
        if not content or not book or not char:
            continue
        
        label_id = 0 if label == 'contradict' else 1  # 1 = consistent
        samples.append((book, char, content, label_id))
    
    print(f"Training on {len(samples)} samples")
    
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        
        pbar = tqdm(samples, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for book, char, content, label in pbar:
            # Get embeddings
            stmt_emb = embedder.encode(content).squeeze()
            backstory_emb = memory.get_backstory_embedding(book, char)
            
            if backstory_emb is None:
                backstory_emb = torch.zeros(embedder.embedding_dim, device=device)
            backstory_emb = backstory_emb.to(device)
            
            # Compute similarity
            similarity = torch.nn.functional.cosine_similarity(
                stmt_emb.unsqueeze(0), 
                backstory_emb.unsqueeze(0)
            )
            
            # Forward
            optimizer.zero_grad()
            logits = classifier(
                similarity,
                stmt_emb.unsqueeze(0),
                backstory_emb.unsqueeze(0)
            )
            
            target = torch.tensor([label], device=device)
            loss = criterion(logits, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=-1).item()
            correct += (pred == label)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        acc = correct / len(samples)
        avg_loss = total_loss / len(samples)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2%}")
    
    print("✅ Training complete!")
    return classifier
