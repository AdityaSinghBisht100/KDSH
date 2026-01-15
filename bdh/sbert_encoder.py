"""
SBERT Encoder Module

Provides sentence-level embeddings using Sentence-BERT.
Replaces byte-level tokenization for semantic understanding.
"""
import torch
import torch.nn as nn
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import re


class SBERTEncoder(nn.Module):
    """
    SBERT-based text encoder.
    
    Converts text into a sequence of sentence embeddings.
    Each sentence becomes one vector of dimension `sbert_dim`.
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = 'cuda',
        max_seq_length: int = 256
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        
        # Load SBERT model
        self.encoder = SentenceTransformer(model_name, device=device)
        self.encoder.max_seq_length = max_seq_length
        
        # Get embedding dimension
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
    
    @property
    def dim(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Uses simple regex-based splitting on sentence boundaries.
        """
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # Filter empty strings and clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no sentences found, treat whole text as one
        if not sentences:
            sentences = [text.strip()] if text.strip() else [""]
        
        return sentences
    
    def encode_text(
        self,
        text: str,
        return_tensor: bool = True
    ) -> Union[torch.Tensor, List[List[float]]]:
        """
        Encode a single text into sentence embeddings.
        
        Args:
            text: Input text string
            return_tensor: If True, return torch.Tensor, else list
            
        Returns:
            Embeddings of shape [num_sentences, embedding_dim]
        """
        sentences = self._split_into_sentences(text)
        
        embeddings = self.encoder.encode(
            sentences,
            convert_to_tensor=return_tensor,
            device=self.device,
            show_progress_bar=False
        )
        
        return embeddings
    
    def encode_batch(
        self,
        texts: List[str],
        padding: bool = True,
        max_sentences: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode a batch of texts into sentence embeddings.
        
        Args:
            texts: List of input text strings
            padding: If True, pad to same length
            max_sentences: Maximum number of sentences per text
            
        Returns:
            Embeddings of shape [batch, max_seq, embedding_dim]
        """
        all_embeddings = []
        max_len = 0
        
        for text in texts:
            emb = self.encode_text(text, return_tensor=True)
            
            # Limit sentences if specified
            if max_sentences is not None and emb.shape[0] > max_sentences:
                emb = emb[:max_sentences]
            
            all_embeddings.append(emb)
            max_len = max(max_len, emb.shape[0])
        
        if not padding:
            return all_embeddings
        
        # Pad to same length
        batch_size = len(texts)
        padded = torch.zeros(batch_size, max_len, self.embedding_dim, device=self.device)
        
        for i, emb in enumerate(all_embeddings):
            padded[i, :emb.shape[0], :] = emb
        
        return padded
    
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Forward pass - encode texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Embeddings tensor
        """
        if isinstance(texts, str):
            return self.encode_text(texts).unsqueeze(0)
        return self.encode_batch(texts)
