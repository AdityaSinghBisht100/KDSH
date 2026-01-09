# Copyright 2025 Pathway Technology, Inc.

import torch
import torch.nn as nn
from typing import List, Iterator, Tuple
import math


class ChunkingStrategy:
    """Strategies for chunking long documents."""
    
    @staticmethod
    def sliding_window(
        tokens: List[int],
        chunk_size: int = 512,
        overlap: int = 128
    ) -> List[List[int]]:
        """
        Split tokens into overlapping chunks.
        Args:
            tokens: List of byte tokens
            chunk_size: Size of each chunk
            overlap: Overlap between consecutive chunks
        Returns:
            List of token chunks
        """
        chunks = []
        stride = chunk_size - overlap
        
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + chunk_size]
            if len(chunk) > 0:
                # Pad if necessary
                if len(chunk) < chunk_size:
                    chunk = chunk + [0] * (chunk_size - len(chunk))
                chunks.append(chunk)
            
            # Stop if we've covered all tokens
            if i + chunk_size >= len(tokens):
                break
        
        return chunks
    
    @staticmethod
    def sentence_based(
        sentences: List[str],
        max_tokens_per_chunk: int = 512,
        tokenizer_fn=None
    ) -> List[List[str]]:
        """
        Group sentences into chunks without breaking sentence boundaries.
        Args:
            sentences: List of sentence strings
            max_tokens_per_chunk: Maximum tokens per chunk
            tokenizer_fn: Function to count tokens (if None, uses character count approximation)
        Returns:
            List of sentence chunks
        """
        if tokenizer_fn is None:
            # Approximate: 1 byte per char
            tokenizer_fn = lambda s: list(bytearray(s, 'utf-8'))
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            tokens = tokenizer_fn(sentence)
            sent_length = len(tokens)
            
            if current_length + sent_length > max_tokens_per_chunk and current_chunk:
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = [sentence]
                current_length = sent_length
            else:
                current_chunk.append(sentence)
                current_length += sent_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class EmbeddingCache:
    """Cache for BDH embeddings to avoid recomputation."""
    
    def __init__(self, max_size: int = 10000):
        self.cache: dict = {}
        self.max_size = max_size
        self.access_count: dict = {}  # For LRU eviction
    
    def get(self, key: str) -> torch.Tensor:
        """Get cached embedding."""
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put(self, key: str, value: torch.Tensor):
        """Store embedding in cache."""
        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Find least accessed key
        min_key = min(self.access_count, key=self.access_count.get)
        del self.cache[min_key]
        del self.access_count[min_key]
    
    def clear(self):
        """Clear all cached embeddings."""
        self.cache.clear()
        self.access_count.clear()


class SparseAttentionMask:
    """Generate sparse attention masks for long sequences."""
    
    @staticmethod
    def local_attention_mask(seq_len: int, window_size: int = 128) -> torch.Tensor:
        """
        Create local attention mask where each position attends to window_size neighbors.
        Args:
            seq_len: Sequence length
            window_size: Size of attention window
        Returns:
            [seq_len, seq_len] attention mask (1 = attend, 0 = mask)
        """
        mask = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        
        return mask
    
    @staticmethod
    def strided_attention_mask(seq_len: int, stride: int = 8) -> torch.Tensor:
        """
        Create strided attention mask for hierarchical attention.
        Each position attends to every stride-th position.
        """
        mask = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            # Attend to self
            mask[i, i] = 1
            # Attend to strided positions
            for j in range(0, seq_len, stride):
                mask[i, j] = 1
        
        return mask


class GradientCheckpointing:
    """Utilities for gradient checkpointing to save memory."""
    
    @staticmethod
    def checkpoint_sequential(
        module: nn.Module,
        segments: int,
        input_tensor: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply gradient checkpointing to sequential module.
        Args:
            module: Sequential module to checkpoint
            segments: Number of checkpoint segments
            input_tensor: Input tensor
        Returns:
            Output tensor
        """
        from torch.utils.checkpoint import checkpoint
        
        # Split module into segments
        num_layers = len(list(module.children()))
        layers_per_segment = math.ceil(num_layers / segments)
        
        x = input_tensor
        layers = list(module.children())
        
        for i in range(0, num_layers, layers_per_segment):
            segment_layers = layers[i:i + layers_per_segment]
            segment_module = nn.Sequential(*segment_layers)
            
            # Apply checkpoint to this segment
            x = checkpoint(segment_module, x, use_reentrant=False)
        
        return x


class MixedPrecisionHelper:
    """Helper for mixed precision training."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == 'cuda'
        
        if self.enabled:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
    
    def get_autocast_context(self):
        """Get autocast context for forward pass."""
        if self.enabled:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.amp.autocast(device_type='cuda', dtype=dtype)
        else:
            from contextlib import nullcontext
            return nullcontext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """Step optimizer with gradient scaling."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


class BatchProcessor:
    """Process documents in batches for efficiency."""
    
    @staticmethod
    def batch_iterator(
        items: List,
        batch_size: int
    ) -> Iterator[List]:
        """
        Create batches from list of items.
        Args:
            items: List of items to batch
            batch_size: Size of each batch
        Yields:
            Batches of items
        """
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    @staticmethod
    def collate_variable_length(
        sequences: List[torch.Tensor],
        padding_value: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate sequences of variable length with padding.
        Args:
            sequences: List of 1D tensors
            padding_value: Value to use for padding
        Returns:
            (padded_tensor, lengths_tensor)
        """
        lengths = torch.tensor([len(seq) for seq in sequences])
        max_len = lengths.max().item()
        
        padded = torch.full(
            (len(sequences), max_len),
            padding_value,
            dtype=sequences[0].dtype
        )
        
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        
        return padded, lengths


class MemoryOptimizer:
    """Utilities for memory optimization."""
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'reserved': torch.cuda.memory_reserved() / 1024**2,  # MB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**2,  # MB
            }
        return {}
    
    @staticmethod
    def optimize_model_memory(model: nn.Module):
        """Apply memory optimization to model."""
        # Convert BatchNorm to eval mode to save memory
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
        
        # Clear gradients
        model.zero_grad(set_to_none=True)


def efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: float = 0.0
) -> torch.Tensor:
    """
    Memory-efficient attention implementation.
    Args:
        query: [B, H, T, D]
        key: [B, H, T, D]
        value: [B, H, T, D]
        mask: Optional attention mask
        dropout: Dropout probability
    Returns:
        [B, H, T, D] attention output
    """
    B, H, T, D = query.shape
    
    # Use Flash Attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            dropout_p=dropout if query.requires_grad else 0.0
        )
    
    # Fallback to standard attention
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    
    if dropout > 0 and query.requires_grad:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout)
    
    output = torch.matmul(attn_weights, value)
    return output
