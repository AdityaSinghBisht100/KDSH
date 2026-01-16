"""
Semantic Encoder using E5-Base

Encodes sentences/paragraphs into 768-dim semantic vectors.
Uses intfloat/e5-base-v2 for high-quality sentence embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import numpy as np


class SemanticEncoder(nn.Module):
    """
    Sentence-level semantic encoder using E5-Base.
    
    E5 models require specific prefixes:
    - "query: " for short queries/statements
    - "passage: " for longer documents/backstories
    
    This encoder freezes E5 weights and optionally adds a 
    learned projection layer (768 → target_dim).
    """
    
    def __init__(
        self, 
        target_dim: int = 2048,
        device: str = "cuda",
        use_projection: bool = True
    ):
        super().__init__()
        self.device = device
        self.e5_dim = 768
        self.target_dim = target_dim
        self.use_projection = use_projection
        
        # Load E5 model and tokenizer
        self._load_e5()
        
        # Optional projection layer (768 → target_dim)
        if use_projection and target_dim != self.e5_dim:
            self.projection = nn.Linear(self.e5_dim, target_dim)
            # Initialize with small weights for stability
            nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.projection.bias)
            print(f"SemanticEncoder: Projection {self.e5_dim} → {target_dim}")
        else:
            self.projection = None
            print(f"SemanticEncoder: No projection (output dim={self.e5_dim})")
    
    def _load_e5(self):
        """Load E5-Base model (frozen)."""
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading E5-Base model...")
            self.e5_model = SentenceTransformer('intfloat/e5-base-v2')
            self.e5_model.to(self.device)
            
            # Freeze E5 weights
            for param in self.e5_model.parameters():
                param.requires_grad = False
            
            print("E5-Base loaded and frozen (768-dim embeddings)")
            
        except ImportError:
            print("ERROR: sentence-transformers not installed!")
            raise ImportError("Please install: pip install sentence-transformers")
    
    def encode_text(
        self, 
        texts: Union[str, List[str]], 
        is_query: bool = True,
        normalize: bool = True,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Encode text(s) to semantic vectors.
        
        Args:
            texts: Single string or list of strings
            is_query: If True, add "query: " prefix (for short texts)
                      If False, add "passage: " prefix (for long docs)
            normalize: If True, L2-normalize output vectors
            batch_size: Batch size for encoding
        
        Returns:
            Tensor of shape [n_texts, e5_dim] or [n_texts, target_dim] if projection
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Add E5 prefixes
        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [prefix + t for t in texts]
        
        # Encode with E5 (returns numpy array)
        with torch.no_grad():
            embeddings = self.e5_model.encode(
                prefixed_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_tensor=True,
                device=self.device
            )
        
        # Apply projection if configured
        if self.projection is not None:
            embeddings = self.projection(embeddings)
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def encode_backstory(
        self, 
        backstory: str, 
        chunk_size: int = 400,
        overlap: int = 50
    ) -> torch.Tensor:
        """
        Encode a long backstory by chunking and averaging.
        
        Args:
            backstory: Full backstory text
            chunk_size: Approximate characters per chunk
            overlap: Overlap between chunks
        
        Returns:
            Single embedding vector representing the backstory
        """
        # Split into overlapping chunks
        chunks = []
        start = 0
        while start < len(backstory):
            end = min(start + chunk_size, len(backstory))
            chunk = backstory[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        
        if not chunks:
            chunks = [backstory[:chunk_size] if backstory else "empty"]
        
        # Encode all chunks
        chunk_embeddings = self.encode_text(chunks, is_query=False, normalize=True)
        
        # Average pool and normalize
        avg_embedding = chunk_embeddings.mean(dim=0, keepdim=True)
        avg_embedding = F.normalize(avg_embedding, p=2, dim=-1)
        
        return avg_embedding
    
    def encode_statement(self, statement: str) -> torch.Tensor:
        """Encode a statement (short query)."""
        return self.encode_text(statement, is_query=True, normalize=True)
    
    def compute_similarity(
        self, 
        backstory_emb: torch.Tensor, 
        statement_emb: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between backstory and statement.
        
        Returns:
            Similarity score in [-1, 1] (higher = more aligned)
        """
        # Ensure 2D tensors
        if backstory_emb.dim() == 1:
            backstory_emb = backstory_emb.unsqueeze(0)
        if statement_emb.dim() == 1:
            statement_emb = statement_emb.unsqueeze(0)
        
        # Cosine similarity
        similarity = F.cosine_similarity(backstory_emb, statement_emb, dim=-1)
        return similarity.item()
    
    def forward(
        self, 
        backstory: str, 
        statement: str
    ) -> dict:
        """
        Full forward pass: encode both texts and compute similarity.
        
        Returns:
            Dict with 'backstory_emb', 'statement_emb', 'similarity'
        """
        backstory_emb = self.encode_backstory(backstory)
        statement_emb = self.encode_statement(statement)
        similarity = self.compute_similarity(backstory_emb, statement_emb)
        
        return {
            'backstory_emb': backstory_emb,
            'statement_emb': statement_emb,
            'similarity': similarity
        }


# Global cached encoder (to avoid reloading for each call)
_cached_encoder = None


def get_semantic_encoder(
    target_dim: int = 2048, 
    device: str = "cuda",
    use_projection: bool = True
) -> SemanticEncoder:
    """Get or create cached semantic encoder."""
    global _cached_encoder
    if _cached_encoder is None:
        _cached_encoder = SemanticEncoder(
            target_dim=target_dim,
            device=device,
            use_projection=use_projection
        )
        _cached_encoder.to(device)
    return _cached_encoder
