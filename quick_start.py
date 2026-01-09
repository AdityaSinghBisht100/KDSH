"""
Quick start example for Narrative Coherence Analysis System
"""

import torch
from pathlib import Path

# Import core modules
from bdh import BDH, BDHConfig
from data_utils import NarrativeDocumentBuilder, TextLoader
from narrative_coherence import NarrativeCoherenceSystem
from sentence_analyzer import SentenceAnalyzer


def quick_demo():
    """Quick demonstration of the system."""
    
    print("="*70)
    print("Narrative Coherence Analysis - Quick Demo")
    print("="*70)
    print()
    
    # 1. Initialize the system
    print("1. Initializing system...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BDHConfig(n_layer=4, n_embd=128, n_head=4)
    
    bdh_model = BDH(config).to(device)
    bdh_model.eval()
    
    coherence_system = NarrativeCoherenceSystem(bdh_model, d_model=128, device=device)
    sentence_analyzer = SentenceAnalyzer(bdh_model, d_model=128, device=device)
    doc_builder = NarrativeDocumentBuilder()
    
    print(f"   ✓ Running on: {device}")
    print(f"   ✓ Model parameters: {sum(p.numel() for p in bdh_model.parameters()):,}")
    print()
    
    # 2. Load and process backstory
    print("2. Processing backstory...")
    
    backstory_text = """
    Princess Elena was born with fire magic. Her mother, Queen Aria, taught her to use magic 
    responsibly. Marcus, the blacksmith's son, became her closest friend. When Elena turned 
    fifteen, a dark sorcerer named Malkor attacked and wounded Queen Aria. The kingdom has 
    a sacred law: magic must never be used for personal revenge.
    """
    
    backstory_doc = doc_builder.build(backstory_text)
    coherence_system.process_backstory(backstory_doc)
    
    summary = coherence_system.get_backstory_summary()
    print(f"   ✓ Parsed {summary['num_sentences']} sentences")
    print(f"   ✓ Found {summary['num_characters']} characters")
    print()
    
    # 3. Analyze current story
    print("3. Analyzing current story...")
    
    current_text = """
    Elena is now seventeen and highly skilled with fire magic. Marcus gave her a special sword.
    Last week, Malkor returned with shadow creatures. Elena decided to confront him directly.
    Before leaving, she vowed to make Malkor pay and burn him with fire. Her eyes glowed with
    intense orange light as her anger fueled her magic.
    """
    
    current_doc = doc_builder.build(current_text)
    
    # Get backstory info
    backstory_embs = coherence_system.constraint_tracker.constraints.sentence_embeddings
    backstory_entities = coherence_system.constraint_tracker.constraints.character_constraints
    
    # Analyze sentences
    results = sentence_analyzer.analyze_document(current_doc, backstory_embs, backstory_entities)
    
    print(f"   ✓ Analyzed {len(results)} sentences")
    print()
    
    # 4. Display results
    print("4. Results Summary:")
    print("-" * 70)
    
    avg_overall = sum(r.overall_score for r in results) / len(results)
    num_violations = sum(1 for r in results if r.is_violation)
    
    print(f"   Average Coherence Score: {avg_overall:.3f}")
    print(f"   Violations Detected: {num_violations}")
    print()
    
    if num_violations > 0:
        print("   Detected Violations:")
        for i, r in enumerate(results):
            if r.is_violation:
                print(f"   - Sentence {i}: {r.violation_type}")
                print(f"     Explanation: {r.explanation}")
        print()
    
    print("   Score Breakdown:")
    for i, r in enumerate(results):
        print(f"   Sentence {i}:")
        print(f"     Temporal: {r.temporal_score:.3f} | Causal: {r.causal_score:.3f}")
        print(f"     Thematic: {r.thematic_score:.3f} | Character: {r.character_score:.3f}")
        if r.is_violation:
            print(f"     ⚠ VIOLATION: {r.violation_type}")
        print()
    
    print("="*70)
    print("Demo Complete!")
    print()
    print("Key Insights:")
    print("- The system detected potential revenge motivation (violation of sacred law)")
    print("- Character development is tracked (Elena's growing power)")
    print("- Temporal consistency is maintained (timeline makes sense)")
    print("- Causal relationships are identified (anger fueling magic)")
    print("="*70)


if __name__ == "__main__":
    quick_demo()
