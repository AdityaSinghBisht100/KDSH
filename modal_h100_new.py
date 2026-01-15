"""
Modal H100 Deployment - New BDH Architecture

Uses pretrained embeddings + semantic consistency.
"""
import modal
import os

app = modal.App("bdh-semantic-consistency")
model_volume = modal.Volume.from_name("bdh-model-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "pandas",
        "tqdm",
        "sentence-transformers",  # For pretrained embeddings
        "transformers"  # Dependency
    )
    .add_local_dir(".", remote_path="/root/bdh_workspace")
)

@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/root/models": model_volume},
)
def run_pipeline():
    import sys
    import torch
    import pandas as pd
    
    sys.path.insert(0, "/root/bdh_workspace")
    os.chdir("/root/bdh_workspace")
    
    from bdh.config import BDHConfig
    from bdh.embeddings import SemanticEmbedder
    from bdh.memory import CharacterMemory
    from pipeline.ingestion import ingest_novels
    from pipeline.training import train_consistency_model, ConsistencyClassifier
    from pipeline.inference import generate_predictions
    
    print(f"🚀 Running on {torch.cuda.get_device_name(0)}")
    
    # Configuration
    config = BDHConfig()
    DATA_DIR = "./files"
    MODEL_DIR = "/root/models"
    
    # Initialize components
    print("\n=== Initializing Components ===")
    embedder = SemanticEmbedder(config.embedding_model, config.device)
    memory = CharacterMemory(config.embedding_dim, config.n_heads, config.device)
    
    # Load training data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Check for cached memory
    cache_path = os.path.join(MODEL_DIR, "character_memory.pt")
    if os.path.exists(cache_path):
        print(f"📦 Loading cached memory from {cache_path}")
        memory.load(cache_path)
    else:
        # Phase 1: Ingest novels
        memory = ingest_novels(
            embedder,
            memory,
            DATA_DIR,
            pd.concat([train_df, test_df]),
            chunk_size=config.chunk_size,
            device=config.device
        )
        
        # Save memory
        memory.save(cache_path)
        model_volume.commit()
        print(f"💾 Memory saved to {cache_path}")
    
    # Phase 2: Train classifier
    classifier_path = os.path.join(MODEL_DIR, "consistency_classifier.pt")
    if os.path.exists(classifier_path):
        print(f"📦 Loading trained classifier from {classifier_path}")
        classifier = ConsistencyClassifier(config.embedding_dim).to(config.device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=config.device))
    else:
        classifier = train_consistency_model(
            embedder,
            memory,
            train_df,
            device=config.device,
            epochs=10,
            lr=1e-3
        )
        
        # Save classifier
        torch.save(classifier.state_dict(), classifier_path)
        model_volume.commit()
        print(f"💾 Classifier saved to {classifier_path}")
    
    # Phase 3: Generate predictions
    output_path = "/root/bdh_workspace/submission_h100.csv"
    result_df = generate_predictions(
        embedder,
        memory,
        classifier,
        test_df,
        output_path,
        device=config.device
    )
    
    # Read for return
    with open(output_path, 'r') as f:
        csv_content = f.read()
    
    return csv_content

@app.local_entrypoint()
def main():
    print("Triggering H100 Job...")
    csv_result = run_pipeline.remote()
    
    if csv_result:
        local_output = "submission_modal.csv"
        with open(local_output, "w") as f:
            f.write(csv_result)
        print(f"✅ Done! Saved to {local_output}")
    else:
        print("❌ Job failed.")
