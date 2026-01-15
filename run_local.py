"""
Local Runner for BDH

Test the BDH Dragon Hatchling architecture locally before deploying to Modal.
"""
import os
import sys
import torch
import pandas as pd
import glob

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from bdh import BDH_GPU, BDHConfig, CONFIGS, ByteTokenizer
from pipeline import pretrain_on_novels, train_consistency_classifier, generate_predictions


def main():
    print("🐉 BDH Dragon Hatchling - Local Test")
    
    # Use tiny config for local testing
    config = CONFIGS["tiny"]
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {config.device}")
    
    DATA_DIR = "./files"
    MODEL_DIR = "./models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Find novels
    novel_paths = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    print(f"📚 Found {len(novel_paths)} novels")
    
    if len(novel_paths) == 0:
        print("❌ No novels found in ./files/")
        return
    
    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"📊 Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Initialize model
    model = BDH_GPU(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {n_params:,} parameters")
    
    # Quick test: forward pass
    print("\n🔬 Testing forward pass...")
    tokenizer = ByteTokenizer()
    test_text = "Hello, this is a test of the BDH model."
    tokens = tokenizer.encode(test_text)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    if config.device == "cuda":
        model = model.cuda()
        input_ids = input_ids.cuda()
    
    with torch.no_grad():
        logits, state = model.forward(input_ids)
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   State layers: {len(state)}")
    
    # Phase 1: Pretrain (quick, 1 epoch)
    print("\n🏋️ Phase 1: Quick Pretraining (1 epoch)...")
    pretrain_path = os.path.join(MODEL_DIR, "bdh_pretrained.pt")
    
    if os.path.exists(pretrain_path):
        print(f"Loading cached model from {pretrain_path}")
        model.load_state_dict(torch.load(pretrain_path, map_location=config.device))
    else:
        model = pretrain_on_novels(
            model,
            novel_paths[:1],  # Just first novel for speed
            epochs=1,
            batch_size=1,
            lr=1e-4,
            device=config.device,
            save_path=pretrain_path
        )
    
    # Phase 2: Train classifier
    print("\n🏋️ Phase 2: Training Classifier...")
    model, classifier = train_consistency_classifier(
        model,
        train_df.head(20),  # Small subset for testing
        DATA_DIR,
        epochs=3,
        batch_size=2,
        lr=1e-3,
        device=config.device
    )
    
    # Phase 3: Predict
    print("\n🔮 Phase 3: Generating Predictions...")
    output_path = "./predictions_local.csv"
    result_df = generate_predictions(
        model,
        classifier,
        test_df.head(10),  # Small subset for testing
        DATA_DIR,
        output_path,
        device=config.device
    )
    
    print(f"\n✅ Complete! Results saved to {output_path}")
    print("\nSample predictions:")
    print(result_df.head(10))


if __name__ == "__main__":
    main()
