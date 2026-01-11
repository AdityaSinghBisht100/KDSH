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
    
    args = parser.parse_args()
    
    print("Initializing KDSH System...")
    
    # Initialize system (local execution)
    system = NarrativeConsistencySystem()
    
    if args.verify:
        print("Running verification...")
        # We need to manually trigger what verify_pipeline does, since we are not in Modal context potentially
        # Or we can just call the method if we instantiate it.
        # Note: calling modal methods locally requires creating an instance.
        
        # We need to use 'with' or manual init
        with system as s:
            s.verify_pipeline.local(s) 
            # Note: .local() is for Modal functions. If it's a class method decorated with @modal.method(),
            # calling it directly on an instance might work if we aren't using the Modal runner stub.
            # But the class definition in system.py uses @app.cls.
            # Local instantiation of @app.cls decorated class usually works as a normal class 
            # but methods might need care.
            
            pass 
            
    else:
        print("System ready. Use --verify to run test.")
        
if __name__ == "__main__":
    main()
