
import torch
import sys
import os

# Ensure we can import from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from narrative_system import NarrativeConsistencySystem
from narrative_system.consistency import CounterfactualChecker

def debug_consistency():
    print("=== Debugging Consistency Logic ===")
    
    # Initialize System
    system = NarrativeConsistencySystem()
    system._initialize_components()
    
    # Create a Dummy World State (Simulating that we know "The sky is blue")
    print("\n1. Creating Dummy World State...")
    system.bdh.reset_state()
    
    # Teach it "The sky is blue." repeatedly to burn it in
    text = "The sky is blue. " * 10
    tokens = system.tokenizer.encode(text)
    t_tensor = torch.tensor([tokens], dtype=torch.long, device=system.device)
    
    # Run forward to get state
    system.bdh(t_tensor, use_state=True)
    world_state = system.bdh.get_state()
    
    print("State Vector Norm:", torch.norm(world_state).item())
    
    # Initialize Checker
    checker = CounterfactualChecker(system.bdh, system.tokenizer, system.device)
    
    # Test Case
    statement = "The sky is red."
    
    print(f"\n2. Testing Statement: '{statement}'")
    
    # A. Check Negation
    negated = checker.negate(statement)
    print(f"   Negation Generated: '{negated}'")
    
    # B. Check Surprise Values
    surprise_S = checker.compute_surprise(statement, world_state)
    surprise_negS = checker.compute_surprise(negated, world_state)
    
    print(f"   Surprise(Statement): {surprise_S:.4f}")
    print(f"   Surprise(Negation):  {surprise_negS:.4f}")
    
    # C. Ratio
    epsilon = 1e-8
    ratio = surprise_S / (surprise_negS + epsilon)
    print(f"   Conflict Ratio: {ratio:.4f}")
    
    if ratio > 1.0:
        print("   Result: CONTRADICT (Correct behavior for specific context)")
    else:
        print("   Result: CONSISTENT (Possible Error if state was strong)")

if __name__ == "__main__":
    debug_consistency()
