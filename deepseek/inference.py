import torch
import pandas as pd
import os
import sys

# Ensure parent directory is in sys.path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from deepseek.config import config
from deepseek.models.bdh_encoder import BDHTextEncoder
from deepseek.models.consistency_classifier import ConsistencyClassifier
from deepseek.pipeline.bdh_state_manager import BDHStateManager
from deepseek.pipeline.narrative_analyzer import NarrativeAnalyzer
from deepseek.processing.data_loader import DataLoader
from deepseek.processing.text_processor import TextProcessor
from deepseek.processing.constraint_extractor import LearnedConstraintDetector

def run_inference():
    device = config.DEVICE
    loader = DataLoader(config)
    processor = TextProcessor(config)
    
    # Initialize components
    text_encoder = BDHTextEncoder(config)
    constraint_detector = LearnedConstraintDetector(config.HIDDEN_SIZE).to(device)
    state_manager = BDHStateManager(config)
    classifier = ConsistencyClassifier(config.HIDDEN_SIZE).to(device)
    
    analyzer = NarrativeAnalyzer(state_manager, classifier)
    
    # Load data
    test_df = loader.load_test_data()
    novels = loader.load_novels()
    
    # 1. Process novels (simplified)
    for name, text in novels.items():
        # In a real scenario, we'd extract character mentions and update states
        pass
        
    results = []
    print("Running inference...")
    for _, row in test_df.iterrows():
        char_name = row['char']
        content = row['content']
        
        # Get backstory embedding
        backstory_emb = text_encoder.encode_text(content) # [1, 768]
        
        # Analyze
        analysis = analyzer.analyze_consistency(char_name, backstory_emb)
        
        results.append({
            "Story ID": row['id'],
            "Prediction": analysis['prediction'],
            "Rationale": analysis['rationale']
        })
        
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)
    print("Results saved to results.csv")

if __name__ == "__main__":
    run_inference()