"""
Simple test script to verify the narrative coherence system works.
This tests the basic functionality without requiring training data.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bdh import BDH, BDHConfig
from data_utils import TextLoader, NarrativeDocumentBuilder
from narrative_coherence import NarrativeCoherenceSystem
from sentence_analyzer import SentenceAnalyzer


def test_data_loading():
    """Test data loading and preprocessing."""
    print("Testing data loading and preprocessing...")
    
    # Create sample text
    sample_text = "Princess Elena had fire magic. She trained with Marcus. The kingdom was in danger."
    
    # Build document
    builder = NarrativeDocumentBuilder()
    doc = builder.build(sample_text)
    
    assert len(doc.sentences) > 0, "Should parse sentences"
    assert len(doc.entities) > 0, "Should extract entities"
    
    print(f"  ✓ Parsed {len(doc.sentences)} sentences")
    print(f"  ✓ Extracted {len(doc.entities)} entities: {list(doc.entities.keys())}")
    print()


def test_bdh_model():
    """Test BDH model initialization."""
    print("Testing BDH model...")
    
    config = BDHConfig(n_layer=2, n_embd=64, n_head=2)
    device = torch.device('cpu')
    model = BDH(config).to(device)
    model.eval()
    
    # Test forward pass
    test_input = torch.randint(0, 256, (1, 10), device=device)
    with torch.no_grad():
        logits, _ = model(test_input)
    
    assert logits.shape == (1, 10, 256), "Output shape should match"
    
    print(f"  ✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"  ✓ Forward pass successful")
    print()


def test_narrative_coherence():
    """Test narrative coherence system."""
    print("Testing narrative coherence system...")
    
    config = BDHConfig(n_layer=2, n_embd=64, n_head=2)
    device = torch.device('cpu')
    bdh_model = BDH(config).to(device)
    bdh_model.eval()
    
    # Initialize system
    system = NarrativeCoherenceSystem(bdh_model, d_model=64, device=device)
    
    # Create sample backstory
    backstory_text = "Princess Elena was a fire mage. Marcus was her friend. Queen Aria was wounded."
    builder = NarrativeDocumentBuilder()
    backstory_doc = builder.build(backstory_text)
    
    # Process backstory
    system.process_backstory(backstory_doc)
    
    # Get summary
    summary = system.get_backstory_summary()
    
    print(f"  ✓ Processed backstory")
    print(f"  ✓ Found {summary.get('num_sentences', 0)} sentences")
    print(f"  ✓ Tracked {summary.get('num_characters', 0)} characters")
    print()


def test_sentence_analyzer():
    """Test sentence analyzer."""
    print("Testing sentence analyzer...")
    
    config = BDHConfig(n_layer=2, n_embd=64, n_head=2)
    device = torch.device('cpu')
    bdh_model = BDH(config).to(device)
    bdh_model.eval()
    
    # Initialize analyzer
    analyzer = SentenceAnalyzer(bdh_model, d_model=64, device=device)
    
    # Create sample sentence
    builder = NarrativeDocumentBuilder()
    doc = builder.build("Elena fought bravely.")
    sentence = doc.sentences[0]
    
    # Encode sentence
    emb = analyzer.encode_sentence(sentence)
    
    assert emb.shape == (64,), "Embedding should be d_model dimensional"
    
    print(f"  ✓ Encoded sentence successfully")
    print(f"  ✓ Embedding shape: {emb.shape}")
    print()


def test_gnn_import():
    """Test PyTorch Geometric import."""
    print("Testing PyTorch Geometric import...")
    
    try:
        from temporal_causal_gnn import TemporalCausalGNN, NarrativeGraph
        print("  ✓ PyTorch Geometric imported successfully")
        
        # Try to create GNN
        device = torch.device('cpu')
        gnn = TemporalCausalGNN(d_model=64, n_heads=2, dropout=0.1)
        print("  ✓ GNN model created successfully")
        
    except ImportError as e:
        print(f"  ⚠ PyTorch Geometric not installed: {e}")
        print("  Run: pip install torch-geometric")
    print()


def test_full_pipeline():
    """Test full inference pipeline."""
    print("Testing full inference pipeline...")
    
    # Check if example files exist
    backstory_file = Path("examples/example_1_backstory.txt")
    current_file = Path("examples/example_1_current.txt")
    
    if backstory_file.exists() and current_file.exists():
        from inference import NarrativeAnalyzer
        
        # Initialize analyzer
        analyzer = NarrativeAnalyzer(device=torch.device('cpu'), use_cache=False)
        
        # Process backstory
        analyzer.process_backstory(str(backstory_file))
        
        # Analyze current story
        results = analyzer.analyze_current_story(str(current_file))
        
        # Generate report
        report = analyzer.generate_report(results)
        
        print(f"  ✓ Processed {report['summary']['total_sentences']} sentences")
        print(f"  ✓ Detected {report['summary']['violations_detected']} violations")
        print(f"  ✓ Average overall score: {report['average_scores']['overall']:.3f}")
        
    else:
        print("  ⚠ Example files not found, skipping pipeline test")
    
    print()


def main():
    """Run all tests."""
    print("="*70)
    print("NARRATIVE COHERENCE SYSTEM - TEST SUITE")
    print("="*70)
    print()
    
    try:
        test_data_loading()
        test_bdh_model()
        test_narrative_coherence()
        test_sentence_analyzer()
        test_gnn_import()
        test_full_pipeline()
        
        print("="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
