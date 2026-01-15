"""
Semantic Embeddings using Sentence-Transformers

Uses pretrained models that ALREADY understand English semantics.
No training required - just encode text into meaningful vectors.
"""
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List, Union

class SemanticEmbedder:
    """
    Wrapper for Sentence-Transformer embeddings.
    
    These embeddings capture semantic meaning:
    - "The doctor treated patients" ≈ "The physician cured the sick"
    - "John is tall" ≠ "John is short"
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        self.device = device
        print(f"📥 Loading pretrained embeddings: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Embeddings loaded! Dimension: {self.embedding_dim}")
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> torch.Tensor:
        """
        Encode text(s) into semantic vectors.
        
        Args:
            texts: Single string or list of strings
            normalize: If True, normalize to unit vectors (for cosine similarity)
            
        Returns:
            torch.Tensor of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts, 
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            device=self.device
        )
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Returns:
            Float in [-1, 1], higher = more similar
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return torch.nn.functional.cosine_similarity(emb1, emb2).item()
    
    def batch_similarity(self, query: str, candidates: List[str]) -> torch.Tensor:
        """
        Compute similarity between query and multiple candidates.
        """
        query_emb = self.encode(query)
        cand_emb = self.encode(candidates)
        return torch.nn.functional.cosine_similarity(query_emb, cand_emb)
