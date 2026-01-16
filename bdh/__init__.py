"""
BDH Package

Implementation of the Dragon Hatchling (BDH) architecture.
Paper: "The Dragon Hatchling: The Missing Link Between The Transformer And Models of the Brain"
arXiv:2509.26507v1
"""
from .config import BDHConfig, CONFIGS
from .model import BDH_GPU, BDHBlock, BDHForConsistency
from .tokenizer import ByteTokenizer, text_to_bytes, bytes_to_text
from .layers import RoPE, BDHLayerNorm, LinearAttention
from .semantic_encoder import SemanticEncoder, get_semantic_encoder

__all__ = [
    # Config
    "BDHConfig",
    "CONFIGS",
    # Model
    "BDH_GPU",
    "BDHBlock", 
    "BDHForConsistency",
    # Tokenizer
    "ByteTokenizer",
    "text_to_bytes",
    "bytes_to_text",
    # Layers
    "RoPE",
    "BDHLayerNorm",
    "LinearAttention",
    # Semantic
    "SemanticEncoder",
    "get_semantic_encoder",
]

