"""
Download pretrained model files from Google Drive to Modal volume.

Usage:
    modal run download_models.py

Files downloaded:
    - narrative_consistency.pt
    - bdh_base.pt
    - world_state_cache.pt (if available)
"""
import modal

# Define the Modal App
app = modal.App("bdh-download-models")

# Define the Volume (same as used in modal_h100.py)
model_volume = modal.Volume.from_name("bdh-model-vol", create_if_missing=True)

# Image with gdown for Google Drive downloads
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("gdown")
)

@app.function(
    image=image,
    timeout=1800,
    volumes={"/root/models": model_volume},
)
def download_from_gdrive():
    import gdown
    import os
    
    MODEL_DIR = "/root/models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Google Drive folder ID from the shared link
    # https://drive.google.com/drive/folders/1E1iVPSH7ELFddX09CXfL-9kJIpe6VoTc
    FOLDER_ID = "1E1iVPSH7ELFddX09CXfL-9kJIpe6VoTc"
    
    # Files to download (with their Google Drive file IDs if known, or we download entire folder)
    FILES = [
        "narrative_consistency.pt",
        "bdh_base.pt",
        "world_state_cache.pt",
        "bdh_transformer.pt"
    ]
    
    print(f"📥 Downloading models from Google Drive folder: {FOLDER_ID}")
    print(f"📁 Target directory: {MODEL_DIR}")
    
    # Method 1: Download the entire folder
    try:
        url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        gdown.download_folder(url, output=MODEL_DIR, quiet=False, use_cookies=False)
        print("✅ Folder download complete!")
    except Exception as e:
        print(f"⚠️ Folder download failed: {e}")
        print("Trying individual file download...")
        
        # Method 2: If folder download fails, try downloading by file IDs
        # (This requires knowing individual file IDs - attempt via fuzzy matching)
        try:
            gdown.download_folder(
                id=FOLDER_ID, 
                output=MODEL_DIR, 
                quiet=False,
                remaining_ok=True
            )
        except Exception as e2:
            print(f"❌ Individual download also failed: {e2}")
            return False
    
    # List downloaded files
    print("\n📋 Files in volume after download:")
    for f in os.listdir(MODEL_DIR):
        filepath = os.path.join(MODEL_DIR, f)
        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"  - {f} ({size:.2f} MB)")
        
    # Commit the volume to persist data
    model_volume.commit()
    print("\n💾 Volume committed successfully!")
    
    return True


@app.local_entrypoint()
def main():
    print("🚀 Starting Google Drive download to Modal volume...")
    success = download_from_gdrive.remote()
    
    if success:
        print("\n✅ All models downloaded and saved to volume 'bdh-model-vol'")
        print("You can now run: modal run modal_h100.py")
    else:
        print("\n❌ Download failed. Check the logs above for details.")
