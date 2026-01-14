import modal
import os
from pathlib import Path

# 1. Define the App
app = modal.App("narrative-consistency-h100")

# 1.1 Define Volume for Persistence
volume = modal.Volume.from_name("narrative-volume", create_if_missing=True)

# 2. Define the Image
# In Modal 1.0, we add local files directly to the Image using add_local_dir
local_project_root = Path(__file__).parent.resolve()

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "transformers>=4.30.0",
        "tqdm",
    )
    # Add project code and data to the image
    .add_local_dir(
        local_project_root,
        remote_path="/root/project",
        # Ignore venv and hidden files
        ignore=lambda p: any(part.startswith('.') for part in Path(p).parts) or "venv" in str(p)
    )
)

@app.function(
    image=image,
    gpu="H100",
    timeout=3600, # 1 hour
    volumes={"/root/project/models": volume}
)
def run_narrative_system(task: str = "verify", book_name: str = None, char_name: str = None, content: str = None):
    import sys
    import torch
    import pandas as pd
    
    # Add project root to sys.path for internal imports
    sys.path.append("/root/project")
    os.chdir("/root/project")
    
    from narrative_system.system import NarrativeConsistencySystem
    
    print(f"🚀 Initializing NarrativeConsistencySystem on {torch.cuda.get_device_name(0)}")
    
    # Initialize system
    system = NarrativeConsistencySystem(data_dir="/root/project/files", model_dir="/root/project/models")
    
    with system:
        if task == "verify":
            print("--- Running Verification ---")
            system.verify_pipeline()
            
        elif task == "train":
            print("--- Starting Training ---")
            train_df = pd.read_csv("/root/project/files/train.csv")
            system.ingest_novel_knowledge(train_df, test_mode=False)
            system.train(epochs=5)
            
        elif task == "predict" and book_name and char_name and content:
            print(f"--- Predicting for {char_name} in {book_name} ---")
            result = system.predict_single(book_name, char_name, content)
            print(f"Result: {result}")
            return result
            
        elif task == "generate_predictions":
            print("--- Generating Predictions for test.csv ---")
            system.generate_predictions(input_file="/root/project/files/test.csv", output_file="/root/project/predictions.csv")
            print("Predictions saved to /root/project/predictions.csv")

@app.local_entrypoint()
def main():
    print("🛰️ Triggering H100 Job on Modal...")
    
    # Switched to "train" to process full book content and all characters
    run_narrative_system.remote(task="train")
