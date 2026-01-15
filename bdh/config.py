"""
BDH Configuration
"""
from dataclasses import dataclass

@dataclass
class BDHConfig:
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence-Transformer model
    embedding_dim: int = 384  # Dimension of embeddings
    
    # State-Space settings
    state_dim: int = 384  # Match embedding dim
    n_heads: int = 4
    decay_alpha: float = 0.95  # State decay rate
    
    # Memory settings
    chunk_size: int = 512  # Characters per chunk for ingestion
    max_entities: int = 100
    
    # Training settings
    learning_rate: float = 1e-4
    consistency_threshold: float = 0.5
    
    # Device
    device: str = "cuda"
