"""
MAIN ENTRY POINT for BDH System with SBERT and GPT-2 Integration.

Usage:
    python main.py --mode train --sbert
    python main.py --mode predict --sbert --with_rationale
"""
import argparse
import os
import torch
import pandas as pd
import glob
from bdh import BDHConfig, BDH_GPU, SBERTEncoder, RationaleDecoder
from pipeline import (
    train_sbert_consistency,
    train_with_rationale,
    generate_sbert_predictions
)

def main():
    parser = argparse.ArgumentParser(description="BDH: Baby Dragon Hatchling (SBERT + GPT-2 Edition)")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"], help="Operation mode")
    
    # Feature flags
    parser.add_argument("--sbert", action="store_true", help="Use SBERT embeddings (Required for this version)")
    parser.add_argument("--with_rationale", action="store_true", help="Enable GPT-2 rationale generation")
    
    # Config
    parser.add_argument("--n_neurons", type=int, default=512, help="BDH generic dimension")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="./files", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Output directory")
    
    args = parser.parse_args()
    
    print("🐉 BDH System Initializing...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Load Config
    config = BDHConfig(
        n_neurons=args.n_neurons,
        use_sbert=True
    )
    
    # Initialize Core Model
    model = BDH_GPU(config).to(device)
    
    # Initialize Rationale Decoder if needed
    decoder = None
    if args.with_rationale:
        print("   Initializing GPT-2 Rationale Decoder...")
        decoder = RationaleDecoder(config).to(device)
    
    # Paths
    train_path = os.path.join(args.data_dir, "train.csv")
    test_path = os.path.join(args.data_dir, "test.csv")
    model_path = os.path.join(models_dir, "bdh_sbert.pt")
    
    # --- TRAINING ---
    if args.mode == "train":
        if not os.path.exists(train_path):
            print(f"❌ Train file not found: {train_path}")
            return
            
        print(f"📚 Loading training data from {train_path}")
        train_df = pd.read_csv(train_path)
        
        if args.with_rationale:
            # Joint Training (BDH + Rationale)
            print("🚀 Starting Joint Training (Consistency + Rationales)...")
            model, decoder = train_with_rationale(
                model=model,
                decoder=decoder,
                train_df=train_df,
                novel_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=1e-4,
                device=device,
                save_path=model_path
            )
        else:
            # Simple Consistency Training
            print("🚀 Starting Consistency Training (SBERT only)...")
            model, _ = train_sbert_consistency(
                model=model,
                train_df=train_df,
                novel_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=1e-4,
                device=device,
                save_path=model_path
            )
            
    # --- PREDICTION ---
    elif args.mode == "predict":
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}. Starting with fresh weights (for testing only).")
        else:
            print(f"📥 Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                if decoder and 'decoder' in checkpoint:
                    decoder.load_state_dict(checkpoint['decoder'])
            else:
                model.load_state_dict(checkpoint)
        
        if not os.path.exists(test_path):
            print(f"❌ Test file not found: {test_path}")
            return
            
        print(f"📚 Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)
        
        # We need a dummy classifier for `predict_with_sbert` if strictly needed,
        # but `generate_sbert_predictions` handles it.
        # Wait, `generate_sbert_predictions` expects a classifier.
        # For this script, let's create a simple one if loading failed or fresh.
        classifier = torch.nn.Sequential(
            torch.nn.Linear(config.n_neurons, config.n_neurons // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(config.n_neurons // 4, 2)
        ).to(device)
        
        if os.path.exists(model_path):
             checkpoint = torch.load(model_path, map_location=device)
             if 'classifier' in checkpoint:
                 classifier.load_state_dict(checkpoint['classifier'])
        
        output_file = os.path.join(args.output_dir, "submission.csv")
        
        generate_sbert_predictions(
            model=model,
            classifier=classifier,
            test_df=test_df,
            novel_dir=args.data_dir,
            output_path=output_file,
            device=device,
            decoder=decoder
        )

if __name__ == "__main__":
    main()
