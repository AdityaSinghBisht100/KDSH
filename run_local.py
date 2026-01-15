"""
Local Runner for Testing

Test the new architecture locally before deploying to Modal.
"""
import os
import sys
import torch
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from bdh.config import BDHConfig
from bdh.embeddings import SemanticEmbedder
from bdh.memory import CharacterMemory
from pipeline.ingestion import ingest_novels
from pipeline.training import train_consistency_model
from pipeline.inference import generate_predictions

def main():
    print("=== BDH Narrative Consistency - Local Test ===")
    
    # Configuration
    config = BDHConfig()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    DATA_DIR = "./files"
    MODEL_DIR = "./models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize
    print("\n1. Initializing components...")
    embedder = SemanticEmbedder(config.embedding_model, config.device)
    memory = CharacterMemory(config.embedding_dim, config.n_heads, config.device)
    
    # Load data
    print("\n2. Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"   Train: {len(train_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    # Ingest
    cache_path = os.path.join(MODEL_DIR, "character_memory.pt")
    if os.path.exists(cache_path):
        print(f"\n3. Loading cached memory...")
        memory.load(cache_path)
    else:
        print("\n3. Ingesting novels...")
        memory = ingest_novels(
            embedder,
            memory,
            DATA_DIR,
            pd.concat([train_df, test_df]),
            chunk_size=config.chunk_size,
            device=config.device
        )
        memory.save(cache_path)
    
    # Train
    print("\n4. Training consistency classifier...")
    classifier = train_consistency_model(
        embedder,
        memory,
        train_df,
        device=config.device,
        epochs=5,  # Fewer epochs for local testing
        lr=1e-3
    )
    
    # Predict
    print("\n5. Generating predictions...")
    output_path = "./predictions_local.csv"
    result_df = generate_predictions(
        embedder,
        memory,
        classifier,
        test_df,
        output_path,
        device=config.device
    )
    
    print(f"\n✅ Complete! Results saved to {output_path}")
    print("\nSample predictions:")
    print(result_df.head(10))

if __name__ == "__main__":
    main()
