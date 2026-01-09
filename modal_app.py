"""
Modal deployment script for Narrative Coherence Analysis System.

This script deploys the narrative coherence analysis system to Modal,
allowing serverless cloud-based analysis of backstory-current story pairs.

Usage:
    # Deploy as a web endpoint
    modal deploy modal_app.py
    
    # Run locally for testing
    modal run modal_app.py
    
    # Serve locally with hot reload
    modal serve modal_app.py

Requirements:
    pip install modal
    modal token new  # Authenticate with Modal
"""

import modal
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

# Create Modal app
app = modal.App("narrative-coherence-analyzer")

# Define the container image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "numpy",
    "torch-geometric>=2.3.0",
)

# Create a volume for model checkpoints (persistent storage)
volume = modal.Volume.from_name("narrative-coherence-models", create_if_missing=True)
MODEL_DIR = "/models"


@dataclass
class AnalysisResult:
    """Result from narrative coherence analysis."""
    total_sentences: int
    violations_detected: int
    violation_rate: float
    average_scores: Dict[str, float]
    violations_by_type: Dict[str, List[Dict]]
    detailed_results: List[Dict]


@app.cls(
    image=image,
    gpu="T4",  # Use T4 GPU for inference (cost-effective)
    timeout=600,  # 10 minute timeout
    volumes={MODEL_DIR: volume},
)
class NarrativeCoherenceService:
    """Modal service class for narrative coherence analysis."""
    
    @modal.enter()
    def initialize(self):
        """Initialize the service when container starts."""
        import torch
        import sys
        from pathlib import Path
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing on device: {self.device}")
        
        # Import local modules (they'll be mounted)
        from bdh import BDH, BDHConfig
        from data_utils import NarrativeDocumentBuilder
        from narrative_coherence import NarrativeCoherenceSystem
        from sentence_analyzer import SentenceAnalyzer
        
        # Initialize BDH model
        self.config = BDHConfig(n_layer=6, n_embd=256, n_head=4)
        self.bdh_model = BDH(self.config).to(self.device)
        
        # Load checkpoint if exists
        checkpoint_path = Path(MODEL_DIR) / "narrative_coherence_latest.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'bdh_model' in checkpoint:
                self.bdh_model.load_state_dict(checkpoint['bdh_model'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        self.bdh_model.eval()
        
        # Initialize coherence system
        self.coherence_system = NarrativeCoherenceSystem(
            self.bdh_model,
            d_model=self.config.n_embd,
            device=self.device
        )
        
        # Initialize sentence analyzer
        self.sentence_analyzer = SentenceAnalyzer(
            self.bdh_model,
            d_model=self.config.n_embd,
            device=self.device
        )
        
        # Document builder
        self.doc_builder = NarrativeDocumentBuilder()
        
        print("Service initialized successfully!")
    
    @modal.method()
    def analyze(self, backstory: str, current_story: str) -> Dict:
        """
        Analyze narrative coherence between backstory and current story.
        
        Args:
            backstory: The backstory text
            current_story: The current story text
            
        Returns:
            Analysis report as dictionary
        """
        import torch
        
        # Parse backstory
        backstory_doc = self.doc_builder.build(backstory)
        print(f"Parsed backstory: {len(backstory_doc.sentences)} sentences")
        
        # Process backstory
        self.coherence_system.process_backstory(backstory_doc)
        
        # Parse current story
        current_doc = self.doc_builder.build(current_story)
        print(f"Parsed current story: {len(current_doc.sentences)} sentences")
        
        # Get backstory info
        backstory_embs = self.coherence_system.constraint_tracker.constraints.sentence_embeddings
        backstory_entities = self.coherence_system.constraint_tracker.constraints.character_constraints
        
        # Analyze current story
        results = self.sentence_analyzer.analyze_document(
            current_doc,
            backstory_embs,
            backstory_entities
        )
        
        # Generate report
        return self._generate_report(results, current_doc)
    
    def _generate_report(self, results: List, current_doc) -> Dict:
        """Generate analysis report from results."""
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
        for i, r in enumerate(results):
            if r.is_violation and r.violation_type:
                if r.violation_type not in violations_by_type:
                    violations_by_type[r.violation_type] = []
                violations_by_type[r.violation_type].append({
                    'sentence_index': i,
                    'sentence_text': current_doc.sentences[i].text,
                    'explanation': r.explanation,
                })
        
        return {
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
                    'sentence_text': current_doc.sentences[i].text,
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
    
    @modal.method()
    def get_backstory_summary(self, backstory: str) -> Dict:
        """
        Process backstory and return summary.
        
        Args:
            backstory: The backstory text
            
        Returns:
            Summary of extracted information
        """
        # Parse and process backstory
        backstory_doc = self.doc_builder.build(backstory)
        self.coherence_system.process_backstory(backstory_doc)
        
        # Get summary
        summary = self.coherence_system.get_backstory_summary()
        
        # Add entity names
        entities = list(self.coherence_system.constraint_tracker.constraints.character_constraints.keys())
        
        return {
            **summary,
            'entities': entities,
            'sentences': [s.text for s in backstory_doc.sentences]
        }


# Create web endpoint for REST API
@app.function(
    image=image,
    gpu="T4",
    timeout=600,
)
@modal.web_endpoint(method="POST")
def analyze_narrative(request: Dict) -> Dict:
    """
    REST API endpoint for narrative analysis.
    
    POST /analyze_narrative
    Body: {
        "backstory": "...",
        "current_story": "..."
    }
    
    Returns: Analysis report
    """
    backstory = request.get("backstory", "")
    current_story = request.get("current_story", "")
    
    if not backstory or not current_story:
        return {"error": "Both 'backstory' and 'current_story' are required"}
    
    # Use the service class
    service = NarrativeCoherenceService()
    return service.analyze.remote(backstory, current_story)


# CLI interface for local testing
@app.local_entrypoint()
def main(
    backstory_file: str = None,
    current_story_file: str = None,
):
    """
    Local entrypoint for testing.
    
    Usage:
        modal run modal_app.py --backstory-file backstory.txt --current-story-file current.txt
    """
    if backstory_file and current_story_file:
        # Read files
        with open(backstory_file, 'r') as f:
            backstory = f.read()
        with open(current_story_file, 'r') as f:
            current_story = f.read()
        
        # Run analysis
        service = NarrativeCoherenceService()
        result = service.analyze.remote(backstory, current_story)
        
        # Print results
        print("\n" + "="*70)
        print("NARRATIVE COHERENCE ANALYSIS REPORT")
        print("="*70)
        print(f"\nTotal Sentences: {result['summary']['total_sentences']}")
        print(f"Violations: {result['summary']['violations_detected']}")
        print(f"Violation Rate: {result['summary']['violation_rate']:.2%}")
        print(f"\nAverage Scores:")
        for key, value in result['average_scores'].items():
            print(f"  {key}: {value:.3f}")
        print("="*70)
        
        # Save detailed report
        with open("modal_analysis_report.json", 'w') as f:
            json.dump(result, f, indent=2)
        print("\nDetailed report saved to: modal_analysis_report.json")
    else:
        # Demo mode
        print("Running in demo mode...")
        
        backstory = """
        Princess Elena had fire magic from birth. Her mother Queen Aria taught her 
        to use magic responsibly. Marcus was her closest friend. The kingdom had a 
        sacred law: magic must never be used for personal revenge.
        """
        
        current_story = """
        Elena is now seventeen and very skilled. Malkor, who wounded her mother, 
        has returned. Elena vowed to burn him with fire as revenge. Her eyes glowed 
        orange with anger as she prepared to attack.
        """
        
        service = NarrativeCoherenceService()
        result = service.analyze.remote(backstory, current_story)
        
        print("\n" + "="*70)
        print("DEMO ANALYSIS RESULTS")
        print("="*70)
        print(f"Violations detected: {result['summary']['violations_detected']}")
        print(f"Overall coherence: {result['average_scores']['overall']:.3f}")
        print("="*70)


# Batch processing endpoint
@app.function(
    image=image,
    gpu="T4",
    timeout=1800,  # 30 minutes for batch
)
def batch_analyze(pairs: List[Dict[str, str]]) -> List[Dict]:
    """
    Batch analyze multiple backstory-current story pairs.
    
    Args:
        pairs: List of {"backstory": "...", "current_story": "..."} dicts
        
    Returns:
        List of analysis reports
    """
    service = NarrativeCoherenceService()
    results = []
    
    for i, pair in enumerate(pairs):
        print(f"Analyzing pair {i+1}/{len(pairs)}...")
        result = service.analyze.remote(pair["backstory"], pair["current_story"])
        results.append(result)
    
    return results


# Scheduled job for continuous monitoring (optional)
@app.function(
    image=image,
    schedule=modal.Cron("0 */6 * * *"),  # Every 6 hours
)
def health_check():
    """Periodic health check to ensure service is working."""
    print("Running health check...")
    
    test_backstory = "Test character existed."
    test_current = "Test character continues to exist."
    
    service = NarrativeCoherenceService()
    result = service.analyze.remote(test_backstory, test_current)
    
    if result and 'summary' in result:
        print("Health check passed!")
        return {"status": "healthy", "timestamp": str(modal.now())}
    else:
        print("Health check failed!")
        return {"status": "unhealthy", "timestamp": str(modal.now())}
