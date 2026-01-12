
import modal
import os
import sys

# Define Modal App
app = modal.App("narrative-consistency-system")

# Define Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "pandas",
        "numpy",
        "tqdm",
        "scikit-learn"
    )
)

# Define Volume for Models
vol = modal.Volume.from_name("narrative-models", create_if_missing=True)

# Mount the package and data
# We mount the current directory's 'narrative_system' package to /root/narrative_system
# And 'files' to /root/files
narrative_pkg = modal.Mount.from_local_dir("narrative_system", remote_path="/root/narrative_system")
bdh_module = modal.Mount.from_local_file("bdh.py", remote_path="/root/bdh.py")
data_mount = modal.Mount.from_local_dir("files", remote_path="/root/files")

@app.function(
    image=image,
    gpu="A10G", # Use A10G or similar
    timeout=3600,
    volumes={"/models": vol},
    mounts=[narrative_pkg, bdh_module, data_mount]
)
def train_remote(epochs: int = 5):
    import sys
    sys.path.append("/root") # Ensure imports work
    
    from narrative_system import NarrativeConsistencySystem
    
    print("Starting Remote Training...")
    system = NarrativeConsistencySystem(data_dir="/root/files", model_dir="/models")
    system.train(epochs=epochs)
    print("Remote Training Complete.")

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/models": vol},
    mounts=[narrative_pkg, bdh_module, data_mount]
)
def infer_remote(input_file: str = "test.csv", output_file: str = "results.csv"):
    import sys
    sys.path.append("/root")
    
    from narrative_system import NarrativeConsistencySystem
    
    print(f"Starting Remote Inference on {input_file}...")
    system = NarrativeConsistencySystem(data_dir="/root/files", model_dir="/models")
    system.generate_predictions(input_file=input_file, output_file=output_file)
    
    # Read back the results to return them (optional, or just save to vol/stdout)
    out_path = os.path.join("/root/files", output_file)
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            print(f.read())
            
    print(f"Inference Complete. Results saved to {output_file} (inside container/volume if persisted).")

if __name__ == "__main__":
    # Local entrypoint to trigger remote functions
    # Usage: modal run modal_app.py::train_remote
    print("Run with: modal run modal_app.py::train_remote --epochs 5")
    print("          modal run modal_app.py::infer_remote --input_file test.csv --output_file results.csv")
