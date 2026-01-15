"""
Inference Pipeline

Generate predictions on test data.
"""
import os
import torch
import pandas as pd
from tqdm import tqdm
from typing import Tuple

def predict_consistency(
    embedder,
    memory,
    classifier,
    book: str,
    char: str,
    content: str,
    device: str = "cuda"
) -> Tuple[str, float, str]:
    """
    Predict consistency.
    """
    stmt_emb = embedder.encode(content).squeeze().to(device)
    backstory_emb = memory.get_backstory_embedding(book, char)
    
    if backstory_emb is None:
        backstory_emb = torch.zeros(embedder.embedding_dim, device=device)
    backstory_emb = backstory_emb.to(device)
    
    similarity = torch.nn.functional.cosine_similarity(
        stmt_emb.unsqueeze(0), 
        backstory_emb.unsqueeze(0)
    )
    
    classifier.eval()
    with torch.no_grad():
        logits = classifier(
            similarity,
            stmt_emb.unsqueeze(0),
            backstory_emb.unsqueeze(0)
        )
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()
    
    label = "consistent" if pred == 1 else "contradict"
    score = probs[0, pred].item()
    
    rationale = f"Semantic similarity: {similarity.item():.4f}."
    return label, score, rationale


def generate_predictions(
    embedder,
    memory,
    classifier,
    test_df: pd.DataFrame,
    output_path: str,
    device: str = "cuda"
):
    """
    Generate predictions.
    """
    print("\n=== Phase 3: Generating Predictions ===")
    
    results = []
    classifier.eval()
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            book = str(row.get('book_name', '')).strip()
            char = str(row.get('char', '')).strip()
            content = row.get('content', '')
            
            if pd.isna(content) or not content:
                results.append({
                    'id': row.get('id', 0),
                    'prediction': 1,
                    'rationale': 'Empty content'
                })
                continue
            
            content = str(content).strip()
            
            label, score, rationale = predict_consistency(
                embedder, memory, classifier,
                book, char, content, device
            )
            
            results.append({
                'id': row.get('id', 0),
                'prediction': 1 if label == "consistent" else 0,
                'rationale': rationale
            })
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")
    
    return result_df
