"""
BDH Package

Implementation of the Dragon Hatchling (BDH) architecture.
Paper: "The Dragon Hatchling: The Missing Link Between The Transformer And Models of the Brain"
arXiv:2509.26507v1

Extended with SBERT encoder and GPT-2 rationale decoder.
"""
from .config import BDHConfig, CONFIGS
from .model import BDH_GPU, BDHBlock, BDHForConsistency
from .tokenizer import ByteTokenizer, text_to_bytes, bytes_to_text
from .layers import RoPE, BDHLayerNorm, LinearAttention
from .sbert_encoder import SBERTEncoder
from .decoder import RationaleDecoder, RationaleDecoderWithBDH

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
    # SBERT Encoder
    "SBERTEncoder",
    # Rationale Decoder
    "RationaleDecoder",
    "RationaleDecoderWithBDH",
]

