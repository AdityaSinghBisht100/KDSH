"""
Simple Modal app for testing Narrative Coherence Analysis.
No FastAPI - just runs analysis with local file paths.

Usage:
    modal run modal_app.py --backstory-file examples/example_1_backstory.txt --story-file examples/example_1_current.txt
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("narrative-coherence-test")

# Define the container image with dependencies and local files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy",
        "torch-geometric>=2.3.0",
    )
    .add_local_dir(".", remote_path="/root")
)

# Define Volume for models
vol = modal.Volume.from_name("narrative-models")


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/models": vol}
)
def analyze(backstory_text: str, story_text: str) -> dict:
    """Run narrative coherence analysis."""
    import sys
    sys.path.insert(0, "/root")
    
    import torch
    from bdh import BDH, BDHConfig
    from data_utils import NarrativeDocumentBuilder
    from narrative_coherence import NarrativeCoherenceSystem
    from sentence_analyzer import SentenceAnalyzer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # Initialize BDH
    config = BDHConfig(n_layer=6, n_embd=256, n_head=4)
    model = BDH(config).to(device)
    
    # Load base BDH weights for consistency
    bdh_ckpt = Path("/models/bdh_base.pt")
    if bdh_ckpt.exists():
        try:
            model.load_state_dict(torch.load(bdh_ckpt, map_location=device))
            print("Loaded BDH weights from /models/bdh_base.pt")
        except Exception as e:
            print(f"Failed to load BDH weights: {e}")
            
    model.eval()
    print("BDH model initialized")
    
    # Initialize components
    doc_builder = NarrativeDocumentBuilder()
    coherence_system = NarrativeCoherenceSystem(model, d_model=config.n_embd, device=device)
    doc_builder = NarrativeDocumentBuilder()
    coherence_system = NarrativeCoherenceSystem(model, d_model=config.n_embd, device=device)
    sentence_analyzer = SentenceAnalyzer(model, d_model=config.n_embd, device=device)
    
    # Set to eval mode for inference
    coherence_system.eval()
    sentence_analyzer.eval()
    
    # Load trained Consistency Classifier weights
    consistency_ckpt = Path("/models/narrative_consistency.pt")
    if consistency_ckpt.exists():
        try:
            state_dict = torch.load(consistency_ckpt, map_location=device)
            sentence_analyzer.coherence_scorer.consistency_net.load_state_dict(state_dict)
            print(f"Loaded trained Consistency Classifier from {consistency_ckpt}")
        except Exception as e:
             print(f"Failed to load checkpoint: {e}")
    else:
        print(f"Warning: Checkpoint {consistency_ckpt} not found. Using initialized weights.")
    
    # Process backstory
    print("\n" + "="*60)
    print("PROCESSING BACKSTORY")
    print("="*60)
    backstory_doc = doc_builder.build(backstory_text)
    print(f"Parsed {len(backstory_doc.sentences)} sentences")
    print(f"Found entities: {list(backstory_doc.entities.keys())}")
    coherence_system.process_backstory(backstory_doc)
    summary = coherence_system.get_backstory_summary()
    print(f"Backstory summary: {summary}")
    
    # Process current story
    print("\n" + "="*60)
    print("PROCESSING CURRENT STORY")
    print("="*60)
    current_doc = doc_builder.build(story_text)
    print(f"Parsed {len(current_doc.sentences)} sentences")
    
    # Get backstory info for comparison
    backstory_embs = coherence_system.constraint_tracker.constraints.sentence_embeddings
    backstory_entities = coherence_system.constraint_tracker.constraints.character_constraints
    
    # Analyze
    print("\n" + "="*60)
    print("ANALYZING COHERENCE")
    print("="*60)
    results = sentence_analyzer.analyze_document(
        current_doc,
        backstory_embs,
        backstory_entities
    )
    
    # Generate report
    num_sentences = len(results)
    num_violations = sum(1 for r in results if r.is_violation)
    
    avg_temporal = sum(r.temporal_score for r in results) / num_sentences if num_sentences else 0
    avg_causal = sum(r.causal_score for r in results) / num_sentences if num_sentences else 0
    avg_thematic = sum(r.thematic_score for r in results) / num_sentences if num_sentences else 0
    avg_character = sum(r.character_score for r in results) / num_sentences if num_sentences else 0
    avg_overall = sum(r.overall_score for r in results) / num_sentences if num_sentences else 0
    avg_relevance = sum(r.relevance_to_backstory for r in results) / num_sentences if num_sentences else 0
    
    # Print violations
    print("\n" + "="*60)
    print("VIOLATIONS DETECTED")
    print("="*60)
    for i, r in enumerate(results):
        if r.is_violation:
            print(f"\nSentence {i}: {current_doc.sentences[i].text[:100]}...")
            print(f"  Type: {r.violation_type}")
            print(f"  Explanation: {r.explanation}")
    
    if num_violations == 0:
        print("No violations detected!")
    
    return {
        'total_sentences': num_sentences,
        'violations_detected': num_violations,
        'violation_rate': num_violations / num_sentences if num_sentences else 0,
        'avg_temporal': avg_temporal,
        'avg_causal': avg_causal,
        'avg_thematic': avg_thematic,
        'avg_character': avg_character,
        'avg_overall': avg_overall,
        'avg_relevance': avg_relevance,
    }


@app.local_entrypoint()
def main(
    backstory_file: str = "examples/example_1_backstory.txt",
    story_file: str = "examples/example_1_current.txt",
):
    """
    Run narrative coherence analysis on local files.
    
    Usage:
        modal run modal_app.py
        modal run modal_app.py --backstory-file path/to/backstory.txt --story-file path/to/story.txt
    """
    # Read files
    print(f"Reading backstory from: {backstory_file}")
    with open(backstory_file, 'r', encoding='utf-8') as f:
        backstory_text = f.read()
    
    print(f"Reading story from: {story_file}")
    with open(story_file, 'r', encoding='utf-8') as f:
        story_text = f.read()
    
    print(f"\nBackstory: {len(backstory_text)} chars")
    print(f"Story: {len(story_text)} chars")
    
    # Run analysis on Modal
    print("\nSending to Modal for analysis...")
    result = analyze.remote(backstory_text, story_text)
    
    # Print final results
    print("\n" + "="*70)
    print("NARRATIVE COHERENCE ANALYSIS RESULTS")
    print("="*70)
    print(f"\nTotal Sentences: {result['total_sentences']}")
    print(f"Violations Detected: {result['violations_detected']}")
    print(f"Violation Rate: {result['violation_rate']:.2%}")
    print(f"\nAverage Scores:")
    print(f"  Temporal:  {result['avg_temporal']:.3f}")
    print(f"  Causal:    {result['avg_causal']:.3f}")
    print(f"  Thematic:  {result['avg_thematic']:.3f}")
    print(f"  Character: {result['avg_character']:.3f}")
    print(f"  Overall:   {result['avg_overall']:.3f}")
    print(f"  Relevance: {result['avg_relevance']:.3f}")
    print("="*70)
