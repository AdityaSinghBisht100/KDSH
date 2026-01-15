"""
BDH Configuration

Based on "The Dragon Hatchling" paper (arXiv:2509.26507v1)
Table 2: Model sizes and hyperparameters
"""
from dataclasses import dataclass
import torch

@dataclass
class BDHConfig:
    """
    Configuration for BDH-GPU model.
    
    Default: ~25M parameters (smallest practical size)
    - n = 32768 (number of neurons/particles)
    - d = 256 (embedding dimension per neuron)
    - L = 4 (number of layers)
    """
    # Model architecture
    n_neurons: int = 32768      # Number of neurons (n in paper)
    embed_dim: int = 256        # Embedding dimension per neuron (d in paper)
    n_layers: int = 4           # Number of BDH layers (L in paper)
    
    # Vocabulary (byte-level, no tokenizer needed)
    vocab_size: int = 256       # UTF-8 bytes
    
    # RoPE parameters
    rope_base: float = 10000.0
    
    # Decay factor for linear attention (U matrix eigenvalues)
    decay_rate: float = 0.99    # Controls how fast old info decays
    
    # Training
    max_seq_len: int = 8192     # Maximum sequence length per batch
    
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    # Entity Tracking via Multi-Head Attention
    n_heads: int = 16           # Number of heads for matrix attention
    
    # SBERT Encoder Configuration
    use_sbert: bool = True                      # Use SBERT embeddings instead of byte tokens
    sbert_model: str = 'all-MiniLM-L6-v2'       # SBERT model name
    sbert_dim: int = 384                        # SBERT embedding dimension
    
    # GPT-2 Decoder Configuration (for rationale generation)
    gpt2_model: str = 'gpt2'                    # GPT-2 model name
    gpt2_dim: int = 768                         # GPT-2 embedding dimension
    rationale_prefix_len: int = 10              # Number of prefix tokens for rationale
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        assert self.n_neurons % self.n_heads == 0, f"n_neurons ({self.n_neurons}) must be divisible by n_heads ({self.n_heads})"
        return self.n_neurons // self.n_heads
    
    @property
    def n_params(self) -> int:
        """Approximate parameter count."""
        # Embedding: vocab_size * n_neurons
        # Per layer: ~4 * n_neurons * embed_dim (for E, Dx, Dy matrices)
        # Output: n_neurons * vocab_size
        emb = self.vocab_size * self.n_neurons
        layers = self.n_layers * 4 * self.n_neurons * self.embed_dim
        out = self.n_neurons * self.vocab_size
        return emb + layers + out


# Preset configurations matching Table 2 in the paper
CONFIGS = {
    "micro": BDHConfig(n_neurons=512, embed_dim=64, n_layers=2, n_heads=8),       # ~1M params
    "tiny": BDHConfig(n_neurons=1024, embed_dim=128, n_layers=2, n_heads=16),     # ~3M params
    "small": BDHConfig(n_neurons=2048, embed_dim=256, n_layers=4, n_heads=32),    # ~12M params
    "medium": BDHConfig(n_neurons=4096, embed_dim=384, n_layers=6, n_heads=64),   # ~50M params
}
