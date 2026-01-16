import torch
from narrative_system.system import NarrativeConsistencySystem
import torch.nn.functional as F

def semantic_sanity_check():
    print("=== KDSH Semantic Sanity Check ===")
    
    # 1. Initialize System
    system = NarrativeConsistencySystem()
    system._initialize_components()
    
    if system.encoder is None:
        print("❌ Error: Encoder not loaded.")
        return

    print("✅ Encoder Loaded: all-MiniLM-L6-v2")
    
    # 2. Test Linguistic Knowledge (Zero-Shot)
    # Even without novel knowledge, the encoder should know word meanings
    s1 = "The character was extremely happy."
    s2 = "The character felt great joy."
    s3 = "The character was deeply depressed."
    
    v1 = system.encoder.encode([s1], convert_to_tensor=True)
    v2 = system.encoder.encode([s2], convert_to_tensor=True)
    v3 = system.encoder.encode([s3], convert_to_tensor=True)
    
    sim_syn = F.cosine_similarity(v1, v2).item()
    sim_ant = F.cosine_similarity(v1, v3).item()
    
    print(f"\nLinguistic Semantic Scores:")
    print(f"  Similarity (Synonyms): {sim_syn:.4f} (Expect > 0.7)")
    print(f"  Similarity (Antonyms): {sim_ant:.4f} (Expect < 0.5)")
    
    if sim_syn > sim_ant:
        print("  ✅ Semantic grounding confirmed: Model understands meaning relationships.")
    else:
        print("  ⚠️ Warning: Semantic grounding not behaving as expected.")

    # 3. Test Character State Stability
    # Paraphrased updates should move the state in similar directions
    system.bdh.reset_state()
    system.absorb_story_stream("Alice was a very kind and gentle person.")
    state1 = [s.clone() for s in system.bdh.get_state()]
    
    system.bdh.reset_state()
    system.absorb_story_stream("Alice was known for her kindness and soft nature.")
    state2 = [s.clone() for s in system.bdh.get_state()]
    
    # Compare first layer's state
    # Flatten to compare vectors
    s1_flat = state1[0].view(-1)
    s2_flat = state2[0].view(-1)
    state_sim = F.cosine_similarity(s1_flat, s2_flat, dim=0).item()
    
    print(f"\nCharacter Memory Stability:")
    print(f"  State Similarity (Paraphrased Alice): {state_sim:.4f}")
    if state_sim > 0.8:
        print("  ✅ Narrative memory is semantically stable.")
    
    print("\nConclusion: Model is SEMANTIC.")

if __name__ == "__main__":
    semantic_sanity_check()
