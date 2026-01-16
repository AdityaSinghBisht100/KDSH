import argparse
import os
import sys

# Add the parent directory to sys.path to allow absolute imports from the 'deepseek' package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    parser = argparse.ArgumentParser(description="BDH Narrative Consistency System")
    parser.add_argument("--mode", choices=["train", "inference", "both"], default="both")
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--test_csv", default="data/test.csv")
    parser.add_argument("--output_dir", default="results")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    if args.mode in ["train", "both"]:
        print("=== Training ===")
        from deepseek.train import train
        train()
    
    if args.mode in ["inference", "both"]:
        print("\n=== Inference ===")
        from deepseek.inference import run_inference
        run_inference()
    
    print("\n=== Complete ===")

if __name__ == "__main__":
    main()