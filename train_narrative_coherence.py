# Copyright 2025 Pathway Technology, Inc.

"""
Training script for narrative coherence analysis system.
"""

import os
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from bdh import BDH, BDHConfig
from data_utils import TextLoader, NarrativeDocumentBuilder
from narrative_coherence import NarrativeCoherenceSystem
from sentence_analyzer import SentenceAnalyzer
from optimization import MixedPrecisionHelper, MemoryOptimizer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training configuration
BDH_CONFIG = BDHConfig(
    n_layer=6,
    n_embd=256,
    dropout=0.1,
    n_head=4,
    mlp_internal_dim_multiplier=128,
    vocab_size=256
)

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
MAX_EPOCHS = 10
LOG_FREQ = 10
CHECKPOINT_DIR = "checkpoints"

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class CoherenceTrainer:
    """Trainer for narrative coherence system."""
    
    def __init__(self, config: BDHConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize BDH model
        self.bdh_model = BDH(config).to(device)
        
        # Initialize narrative coherence system
        self.coherence_system = NarrativeCoherenceSystem(
            self.bdh_model,
            d_model=config.n_embd,
            device=device
        )
        
        # Initialize sentence analyzer
        self.sentence_analyzer = SentenceAnalyzer(
            self.bdh_model,
            d_model=config.n_embd,
            device=device
        )
        
        # Document builder
        self.doc_builder = NarrativeDocumentBuilder()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.coherence_system.parameters()) + 
            list(self.sentence_analyzer.parameters()),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Mixed precision helper
        self.mp_helper = MixedPrecisionHelper(device)
    
    def compute_coherence_loss(
        self,
        coherence_scores: dict,
        target_coherence: float = 0.8
    ) -> torch.Tensor:
        """
        Compute loss for coherence scores.
        We want high coherence (close to 1) and low violation probability.
        """
        # Coherence loss (want scores close to target)
        temporal_loss = F.mse_loss(coherence_scores['temporal'], 
                                   torch.tensor(target_coherence, device=self.device))
        causal_loss = F.mse_loss(coherence_scores['causal'], 
                                torch.tensor(target_coherence, device=self.device))
        thematic_loss = F.mse_loss(coherence_scores['thematic'], 
                                   torch.tensor(target_coherence, device=self.device))
        character_loss = F.mse_loss(coherence_scores['character'], 
                                    torch.tensor(target_coherence, device=self.device))
        
        # Violation loss (want low violation probability)
        violation_loss = coherence_scores['violation_prob']  # Already 0-1, want it low
        
        # Combine losses
        total_loss = (
            temporal_loss * 0.25 +
            causal_loss * 0.25 +
            thematic_loss * 0.3 +
            character_loss * 0.2 +
            violation_loss * 0.5  # Higher weight on violation detection
        )
        
        return total_loss
    
    def train_on_pair(
        self,
        backstory_text: str,
        current_story_text: str
    ) -> float:
        """
        Train on a single backstory-current_story pair.
        Returns training loss.
        """
        # Parse documents
        backstory_doc = self.doc_builder.build(backstory_text)
        current_doc = self.doc_builder.build(current_story_text)
        
        # Process backstory
        self.coherence_system.process_backstory(backstory_doc)
        
        # Get backstory embeddings
        backstory_embs = self.coherence_system.constraint_tracker.constraints.sentence_embeddings
        backstory_entities = self.coherence_system.constraint_tracker.constraints.character_constraints
        
        # Train on current story sentences
        total_loss = 0.0
        num_sentences = min(len(current_doc.sentences), 20)  # Limit for memory
        
        for i in range(num_sentences):
            sentence = current_doc.sentences[i]
            
            # Get sentence embedding
            sent_emb = self.sentence_analyzer.encode_sentence(sentence)
            
            # Retrieve backstory context
            backstory_context = self.sentence_analyzer.retrieve_relevant_backstory(
                sent_emb, backstory_embs, top_k=5
            )
            
            # Compute coherence scores
            scores = self.sentence_analyzer.coherence_scorer(sent_emb, backstory_context)
            
            # Compute loss
            loss = self.compute_coherence_loss(scores)
            total_loss += loss
        
        avg_loss = total_loss / num_sentences if num_sentences > 0 else total_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        avg_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            list(self.coherence_system.parameters()) + 
            list(self.sentence_analyzer.parameters()),
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        return avg_loss.item()
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"narrative_coherence_epoch_{epoch}.pt")
        
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'bdh_model': self.bdh_model.state_dict(),
            'coherence_system': self.coherence_system.state_dict(),
            'sentence_analyzer': self.sentence_analyzer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")


def load_training_data(data_dir: str = "training_data"):
    """
    Load training data pairs from directory.
    Expected structure:
        training_data/
            pair_1_backstory.txt
            pair_1_current.txt
            pair_2_backstory.txt
            pair_2_current.txt
            ...
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Warning: Training data directory '{data_dir}' not found")
        return []
    
    # Find all backstory files
    backstory_files = sorted(data_path.glob("*_backstory.txt"))
    
    pairs = []
    for backstory_file in backstory_files:
        # Find corresponding current story file
        prefix = backstory_file.stem.replace("_backstory", "")
        current_file = data_path / f"{prefix}_current.txt"
        
        if current_file.exists():
            backstory_text = TextLoader.load_text(str(backstory_file))
            current_text = TextLoader.load_text(str(current_file))
            pairs.append((backstory_text, current_text))
    
    return pairs


def main():
    """Main training loop."""
    
    # Initialize trainer
    trainer = CoherenceTrainer(BDH_CONFIG, device)
    
    # Load training data
    print("Loading training data...")
    training_pairs = load_training_data()
    
    if len(training_pairs) == 0:
        print("No training data found. Creating example files...")
        print("\nTo train the model, create pairs of files in 'training_data/' directory:")
        print("  - pair_1_backstory.txt")
        print("  - pair_1_current.txt")
        print("  - pair_2_backstory.txt")
        print("  - pair_2_current.txt")
        print("  etc.")
        return
    
    print(f"Found {len(training_pairs)} training pairs")
    
    # Training loop
    for epoch in range(MAX_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
        print(f"{'='*60}")
        
        epoch_loss = 0.0
        
        for pair_idx, (backstory, current) in enumerate(training_pairs):
            # Train on this pair
            loss = trainer.train_on_pair(backstory, current)
            epoch_loss += loss
            
            if (pair_idx + 1) % LOG_FREQ == 0 or pair_idx == len(training_pairs) - 1:
                avg_loss = epoch_loss / (pair_idx + 1)
                print(f"  Pair {pair_idx + 1}/{len(training_pairs)} - Loss: {loss:.4f} (Avg: {avg_loss:.4f})")
            
            # Clear cache periodically
            if (pair_idx + 1) % 10 == 0:
                MemoryOptimizer.clear_cache()
        
        avg_epoch_loss = epoch_loss / len(training_pairs) if len(training_pairs) > 0 else 0
        print(f"\nEpoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        trainer.save_checkpoint(epoch + 1, avg_epoch_loss)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
