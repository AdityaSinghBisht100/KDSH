import os
import modal

# Define the Modal App
app = modal.App("bdh-narrative-consistency")

# Define the H100 Volume to persist the Checkpoint
model_volume = modal.Volume.from_name("bdh-model-vol", create_if_missing=True)

# Define the environment (Container Image)
# Copy all local files into the image at build time
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "pandas", 
        "tqdm",
        "numpy"
    )
    .copy_local_dir(".", "/root/bdh", ignore=[".git", "__pycache__", "*.pt", "models/"])
)

@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/root/models": model_volume},
)
def run_pipeline_on_h100():
    import sys
    import torch
    
    sys.path.append("/root/bdh")
    os.chdir("/root/bdh")
    
    print(f"🚀 Running on Modal! GPU: {torch.cuda.get_device_name(0)}")
    
    from narrative_system import NarrativeConsistencySystem
    from narrative_system.ingestion import ingest_novel_knowledge
    from narrative_system.inference import generate_predictions
    
    DATA_DIR = "./files"
    MODEL_DIR = "/root/models"
    CACHE_FILE = os.path.join(MODEL_DIR, "world_state_h100.pt")
    OUTPUT_FILE = "/root/bdh/submission_h100.csv"
    
    system = NarrativeConsistencySystem(data_dir=DATA_DIR, model_dir=MODEL_DIR)
    system._initialize_components()
    
    if os.path.exists(CACHE_FILE):
        print(f"📦 Found cached state: {CACHE_FILE}")
        checkpoint = torch.load(CACHE_FILE)
        system.world_states = checkpoint['world_states']
        system.backstory_states = checkpoint['backstory_states']
    else:
        print("📖 No cache. Ingesting novels...")
        import pandas as pd
        
        train_path = os.path.join(DATA_DIR, "train.csv")
        test_path = os.path.join(DATA_DIR, "test.csv")
        
        df_list = []
        if os.path.exists(train_path): df_list.append(pd.read_csv(train_path))
        if os.path.exists(test_path): df_list.append(pd.read_csv(test_path))
        
        if df_list:
            combined_df = pd.concat(df_list)
            ingest_novel_knowledge(system, combined_df, test_mode=False)
            
            print(f"💾 Saving to Volume: {CACHE_FILE}")
            torch.save({
                'world_states': system.world_states,
                'backstory_states': system.backstory_states
            }, CACHE_FILE)
            model_volume.commit()
        else:
            print("❌ No CSV files!")
            return None

    print("🧠 Running Inference...")
    generate_predictions(system, input_file="test.csv", output_file=OUTPUT_FILE)
    
    with open(OUTPUT_FILE, "r") as f:
        csv_content = f.read()
        
    return csv_content

@app.local_entrypoint()
def main():
    print("Triggering H100 Job...")
    csv_result = run_pipeline_on_h100.remote()
    
    if csv_result:
        local_output = "submission_modal.csv"
        with open(local_output, "w") as f:
            f.write(csv_result)
        print(f"✅ Done! Saved to {local_output}")
    else:
        print("❌ Job failed.")

