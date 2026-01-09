# Copyright 2025 Pathway Technology, Inc.

"""
Inference pipeline for narrative coherence analysis.
"""

import torch
from typing import List, Dict
import json
from pathlib import Path

from bdh import BDH, BDHConfig
from data_utils import TextLoader, NarrativeDocumentBuilder
from narrative_coherence import NarrativeCoherenceSystem
from sentence_analyzer import SentenceAnalyzer, CoherenceScore
from optimization import EmbeddingCache, MemoryOptimizer


class NarrativeAnalyzer:
    """Main inference interface for narrative coherence analysis."""
    
    def __init__(
        self,
        model_checkpoint: str = None,
        device: torch.device = None,
        use_cache: bool = True
    ):
        """
        Initialize narrative analyzer.
        Args:
            model_checkpoint: Path to trained model checkpoint
            device: Device to run on (defaults to cuda if available)
            use_cache: Whether to use embedding cache
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Initializing Narrative Analyzer on {self.device}")
        
        # Initialize BDH model
        self.bdh_config = BDHConfig()
        self.bdh_model = BDH(self.bdh_config).to(self.device)
        
        # Load checkpoint if provided
        if model_checkpoint and Path(model_checkpoint).exists():
            print(f"Loading checkpoint from {model_checkpoint}")
            self.load_checkpoint(model_checkpoint)
        else:
            print("No checkpoint provided, using untrained model")
        
        self.bdh_model.eval()
        
        # Initialize narrative coherence system
        self.coherence_system = NarrativeCoherenceSystem(
            self.bdh_model,
            d_model=self.bdh_config.n_embd,
            device=self.device
        )
        
        # Initialize sentence analyzer
        self.sentence_analyzer = SentenceAnalyzer(
            self.bdh_model,
            d_model=self.bdh_config.n_embd,
            device=self.device
        )
        
        # Document builder
        self.doc_builder = NarrativeDocumentBuilder()
        
        # Cache
        self.cache = EmbeddingCache() if use_cache else None
        
        # State
        self.backstory_processed = False
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'bdh_model' in checkpoint:
            self.bdh_model.load_state_dict(checkpoint['bdh_model'])
        elif 'model_state_dict' in checkpoint:
            self.bdh_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume entire checkpoint is model state dict
            self.bdh_model.load_state_dict(checkpoint)
        
        print("Checkpoint loaded successfully")
    
    def process_backstory(self, backstory_path: str):
        """
        Load and process backstory text file.
        Args:
            backstory_path: Path to backstory text file
        """
        print(f"\nProcessing backstory from: {backstory_path}")
        
        # Load text
        text = TextLoader.load_text(backstory_path)
        print(f"Loaded {len(text)} characters")
        
        # Build document
        backstory_doc = self.doc_builder.build(text)
        print(f"Parsed {len(backstory_doc.sentences)} sentences")
        print(f"Found {len(backstory_doc.entities)} entities")
        
        # Process with coherence system
        self.coherence_system.process_backstory(backstory_doc)
        
        # Get summary
        summary = self.coherence_system.get_backstory_summary()
        print(f"Backstory summary: {summary}")
        
        self.backstory_processed = True
        
        return backstory_doc
    
    def analyze_current_story(self, current_story_path: str) -> List[CoherenceScore]:
        """
        Analyze current story against processed backstory.
        Args:
            current_story_path: Path to current story text file
        Returns:
            List of coherence scores for each sentence
        """
        if not self.backstory_processed:
            raise RuntimeError("Must process backstory first using process_backstory()")
        
        print(f"\nAnalyzing current story from: {current_story_path}")
        
        # Load and parse current story
        text = TextLoader.load_text(current_story_path)
        print(f"Loaded {len(text)} characters")
        
        current_doc = self.doc_builder.build(text)
        print(f"Parsed {len(current_doc.sentences)} sentences")
        
        # Get backstory embeddings and constraints
        backstory_embs = self.coherence_system.constraint_tracker.constraints.sentence_embeddings
        backstory_entities = self.coherence_system.constraint_tracker.constraints.character_constraints
        
        # Analyze with sentence analyzer
        print("Analyzing sentence-by-sentence coherence...")
        results = self.sentence_analyzer.analyze_document(
            current_doc,
            backstory_embs,
            backstory_entities
        )
        
        print(f"Analysis complete: {len(results)} sentences analyzed")
        
        return results
    
    def generate_report(
        self,
        results: List[CoherenceScore],
        output_path: str = None
    ) -> Dict:
        """
        Generate comprehensive analysis report.
        Args:
            results: List of coherence scores
            output_path: Optional path to save report as JSON
        Returns:
            Report dictionary
        """
        # Calculate statistics
        num_sentences = len(results)
        num_violations = sum(1 for r in results if r.is_violation)
        
        avg_temporal = sum(r.temporal_score for r in results) / num_sentences
        avg_causal = sum(r.causal_score for r in results) / num_sentences
        avg_thematic = sum(r.thematic_score for r in results) / num_sentences
        avg_character = sum(r.character_score for r in results) / num_sentences
        avg_overall = sum(r.overall_score for r in results) / num_sentences
        avg_relevance = sum(r.relevance_to_backstory for r in results) / num_sentences
        
        # Group violations by type
        violations_by_type = {}
        for r in results:
            if r.is_violation and r.violation_type:
                if r.violation_type not in violations_by_type:
                    violations_by_type[r.violation_type] = []
                violations_by_type[r.violation_type].append({
                    'explanation': r.explanation,
                    'temporal_score': r.temporal_score,
                    'causal_score': r.causal_score,
                    'thematic_score': r.thematic_score,
                    'character_score': r.character_score
                })
        
        # Build report
        report = {
            'summary': {
                'total_sentences': num_sentences,
                'violations_detected': num_violations,
                'violation_rate': num_violations / num_sentences if num_sentences > 0 else 0,
            },
            'average_scores': {
                'temporal': float(avg_temporal),
                'causal': float(avg_causal),
                'thematic': float(avg_thematic),
                'character': float(avg_character),
                'overall': float(avg_overall),
                'relevance_to_backstory': float(avg_relevance)
            },
            'violations_by_type': violations_by_type,
            'detailed_results': [
                {
                    'sentence_index': i,
                    'temporal_score': float(r.temporal_score),
                    'causal_score': float(r.causal_score),
                    'thematic_score': float(r.thematic_score),
                    'character_score': float(r.character_score),
                    'overall_score': float(r.overall_score),
                    'relevance': float(r.relevance_to_backstory),
                    'is_violation': r.is_violation,
                    'violation_type': r.violation_type,
                    'explanation': r.explanation
                }
                for i, r in enumerate(results)
            ]
        }
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {output_path}")
        
        return report
    
    def print_summary(self, report: Dict, show_details: bool = True):
        """Print human-readable summary of analysis with detailed output."""
        from tqdm import tqdm
        
        print("\n" + "="*70)
        print("📊 NARRATIVE COHERENCE ANALYSIS REPORT")
        print("="*70)
        
        # Summary section
        print("\n📋 SUMMARY")
        print("-"*40)
        total = report['summary']['total_sentences']
        violations = report['summary']['violations_detected']
        rate = report['summary']['violation_rate']
        
        print(f"  Total Sentences Analyzed: {total}")
        print(f"  Violations Detected:      {violations}")
        print(f"  Violation Rate:           {rate:.2%}")
        
        # Coherence scores with visual bars
        print("\n📈 AVERAGE COHERENCE SCORES")
        print("-"*40)
        
        for key, value in report['average_scores'].items():
            bar_length = int(value * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            label = key.replace('_', ' ').title()
            print(f"  {label:25s} [{bar}] {value:.3f}")
        
        # Violations section
        if report['violations_by_type']:
            print("\n⚠️  VIOLATIONS DETECTED")
            print("-"*40)
            for vtype, violations_list in report['violations_by_type'].items():
                print(f"\n  🔴 {vtype.replace('_', ' ').title()}: {len(violations_list)} occurrences")
                for i, v in enumerate(violations_list[:3]):  # Show first 3 examples
                    print(f"     └─ {v['explanation']}")
        else:
            print("\n✅ NO VIOLATIONS DETECTED")
            print("-"*40)
            print("  All sentences are coherent with the backstory.")
        
        # Detailed sentence-by-sentence results
        if show_details and report['detailed_results']:
            print("\n📝 SENTENCE-BY-SENTENCE ANALYSIS")
            print("-"*40)
            
            for result in report['detailed_results']:
                idx = result['sentence_index']
                overall = result['overall_score']
                is_violation = result['is_violation']
                
                # Status icon
                if is_violation:
                    status = "🔴"
                elif overall < 0.6:
                    status = "🟡"
                else:
                    status = "🟢"
                
                # Score bar
                bar_len = int(overall * 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)
                
                print(f"  {status} Sentence {idx+1:3d}: [{bar}] {overall:.2f}")
                
                # Show violation details
                if is_violation:
                    print(f"       └─ ⚠️  {result['violation_type']}: {result['explanation'][:60]}...")
        
        print("\n" + "="*70)
        print("✨ Analysis Complete!")
        print("="*70)
    
    def print_detailed_sentence(self, result: Dict, sentence_text: str = None):
        """Print detailed analysis for a single sentence."""
        print(f"\n{'─'*50}")
        print(f"Sentence {result['sentence_index'] + 1}")
        if sentence_text:
            print(f"Text: \"{sentence_text[:80]}{'...' if len(sentence_text) > 80 else ''}\"")
        print(f"{'─'*50}")
        
        scores = [
            ("Temporal", result['temporal_score']),
            ("Causal", result['causal_score']), 
            ("Thematic", result['thematic_score']),
            ("Character", result['character_score']),
            ("Overall", result['overall_score']),
        ]
        
        for name, score in scores:
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"  {name:12s}: [{bar}] {score:.3f}")
        
        if result['is_violation']:
            print(f"\n  ⚠️  VIOLATION: {result['violation_type']}")
            print(f"      {result['explanation']}")



def main():
    """Example usage of the narrative analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Narrative Coherence Analysis System - Analyze backstory influence on current story",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --backstory backstory.txt --current current_story.txt
  python inference.py -b backstory.txt -c current_story.txt -o report.json
  python inference.py --backstory backstory.txt --current current_story.txt --checkpoint model.pt
        """
    )
    
    parser.add_argument(
        "-b", "--backstory",
        type=str,
        required=True,
        help="Path to backstory text file"
    )
    
    parser.add_argument(
        "-c", "--current",
        type=str,
        required=True,
        help="Path to current story text file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="coherence_report.json",
        help="Path to output report file (default: coherence_report.json)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint file (optional)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run on (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = None  # Let NarrativeAnalyzer choose
    else:
        device = torch.device(args.device)
    
    # Initialize analyzer
    analyzer = NarrativeAnalyzer(
        model_checkpoint=args.checkpoint,
        device=device,
        use_cache=not args.no_cache
    )
    
    # Process backstory
    analyzer.process_backstory(args.backstory)
    
    # Analyze current story
    results = analyzer.analyze_current_story(args.current)
    
    # Generate and print report
    report = analyzer.generate_report(results, args.output)
    analyzer.print_summary(report)
    
    # Cleanup
    MemoryOptimizer.clear_cache()


if __name__ == "__main__":
    main()

