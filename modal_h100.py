"""
Modal H100 Deployment for BDH

Runs the full BDH training and inference pipeline on H100 GPUs.

Pipeline:
1. Self-Supervised Pretraining: Train BDH on novel text
2. Consistency Classifier Training: Train classifier on labeled data
3. Inference: Generate predictions on test set
"""
import modal
import os

app = modal.App("bdh-dragon-hatchling")
model_volume = modal.Volume.from_name("bdh-model-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "pandas",
        "tqdm",
    )
    .add_local_dir(".", remote_path="/root/bdh_workspace")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,  # 2 hours
    volumes={"/root/models": model_volume},
)
def run_pipeline():
    """
    Full BDH pipeline:
    1. Pretrain on novels
    2. Train consistency classifier
    3. Generate predictions
    """
    import sys
    import torch
    import pandas as pd
    import glob
    
    sys.path.insert(0, "/root/bdh_workspace")
    os.chdir("/root/bdh_workspace")
    
    from bdh import BDH_GPU, BDHConfig, CONFIGS
    from pipeline import pretrain_on_novels, train_consistency_classifier, generate_predictions, contrastive_finetune
    
    print(f"🐉 BDH Dragon Hatchling Pipeline")
    print(f"🚀 Running on {torch.cuda.get_device_name(0)}")
    
    # Configuration - use "small" for better capacity
    config = CONFIGS["small"]  # ~12M params, more capacity for learning
    config.device = "cuda"
    
    DATA_DIR = "./files"
    MODEL_DIR = "/root/models"
    
    # Find novels
    novel_paths = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    print(f"📚 Found {len(novel_paths)} novels")
    
    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"📊 Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Check for cached model
    pretrain_path = os.path.join(MODEL_DIR, "bdh_pretrained_small.pt")
    contrastive_path = os.path.join(MODEL_DIR, "bdh_contrastive_small.pt")
    
    # Initialize model
    model = BDH_GPU(config)
    print(f"🧠 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Phase 1: Pretrain on novels
    if os.path.exists(pretrain_path):
        print(f"📦 Loading pretrained model from {pretrain_path}")
        model.load_state_dict(torch.load(pretrain_path, map_location="cuda"))
    else:
        print("🏋️ Starting pretraining...")
        model = pretrain_on_novels(
            model,
            novel_paths,
            epochs=2,
            batch_size=2,
            lr=1e-4,
            device="cuda",
            save_path=pretrain_path
        )
        model_volume.commit()
    
    # Phase 2: Contrastive fine-tuning (skip old classifier)
    if os.path.exists(contrastive_path):
        print(f"📦 Loading contrastive-finetuned model from {contrastive_path}")
        model.load_state_dict(torch.load(contrastive_path, map_location="cuda"))
    else:
        print("🏋️ Contrastive fine-tuning...")
        model = contrastive_finetune(
            model,
            train_df,
            DATA_DIR,
            epochs=5,
            batch_size=4,
            lr=1e-5,
            margin=1.0,
            device="cuda",
            save_path=contrastive_path
        )
        model_volume.commit()
    
    # Phase 3: Generate predictions
    output_path = "/root/bdh_workspace/submission.csv"
    result_df = generate_predictions(
        model,
        None,  # No classifier, use perplexity-based
        test_df,
        DATA_DIR,
        output_path,
        device="cuda",
        use_classifier=False,
        perplexity_threshold=100.0
    )
    
    # Read result for return
    with open(output_path, 'r') as f:
        csv_content = f.read()
    
    return csv_content


@app.local_entrypoint()
def main():
    """Local entrypoint."""
    print("🐉 Starting BDH Dragon Hatchling Pipeline...")
    csv_result = run_pipeline.remote()
    
    if csv_result:
        local_output = "submission_bdh.csv"
        with open(local_output, "w") as f:
            f.write(csv_result)
        print(f"✅ Done! Saved to {local_output}")
    else:
        print("❌ Job failed.")
