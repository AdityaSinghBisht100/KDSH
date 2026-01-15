# main.py
import argparse
import os

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
        from train import train_model
        train_model()
    
    if args.mode in ["inference", "both"]:
        print("\n=== Inference ===")
        from inference import main as inference_main
        inference_main()
    
    print("\n=== Complete ===")

if __name__ == "__main__":
    main()