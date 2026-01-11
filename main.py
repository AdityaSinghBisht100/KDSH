
import sys
import os
import argparse

# Ensure we can import from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from narrative_system import NarrativeConsistencySystem

def main():
    parser = argparse.ArgumentParser(description="KDSH Narrative Consistency System")
    parser.add_argument("--verify", action="store_true", help="Run verification pipeline")
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--data_dir", type=str, default="./files", help="Directory containing train.csv and book .txt files")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save/load models")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    
    args = parser.parse_args()
    
    print(f"Initializing KDSH System (Data: {args.data_dir}, Models: {args.model_dir})...")
    
    # Initialize system with custom paths
    system = NarrativeConsistencySystem(data_dir=args.data_dir, model_dir=args.model_dir)
    
    if args.verify:
        print("Running verification...")
        # Since verify_pipeline is a modal method, we just call manual logic here or we can call it if it wasn't stripped.
        # But we added it to the class in system.py, so we can call it.
        try:
             system.verify_pipeline()
        except AttributeError:
             # In case Modal wraps it weirdly
             system.verify_pipeline.local(system)
             
    elif args.train:
        print("Running training...")
        system.train(epochs=args.epochs)
        
    else:
        print("System ready.")
        print("Usage:")
        print("  python main.py --verify")
        print("  python main.py --train --data_dir /path/to/data --epochs 10")
        
if __name__ == "__main__":
    main()
