"""
Byte-Level Tokenizer for BDH

The Dragon Hatchling paper uses byte-level tokenization (vocab size 256).
This is simpler than BPE/WordPiece and works universally for any language.
"""
import torch
from typing import List, Union


class ByteTokenizer:
    """
    Simple byte-level tokenizer.
    
    Converts text to bytes and back. No vocabulary to learn.
    Vocab size is always 256 (0-255).
    """
    
    def __init__(self):
        self.vocab_size = 256
        self.pad_token_id = 0  # NULL byte
        self.eos_token_id = 3  # ETX (end of text)
        self.bos_token_id = 2  # STX (start of text)
        print("Using Byte-Level Tokenizer (vocab=256)")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to byte tokens (0-255)."""
        byte_list = list(text.encode('utf-8', errors='replace'))
        if add_special_tokens:
            byte_list = [self.bos_token_id] + byte_list + [self.eos_token_id]
        return byte_list
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode byte tokens back to text."""
        if skip_special_tokens:
            token_ids = [t for t in token_ids 
                        if t not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)]
        return bytes(token_ids).decode('utf-8', errors='replace')
    
    def __call__(
        self, 
        text: Union[str, List[str]], 
        return_tensors: str = None,
        padding: bool = False,
        max_length: int = None,
        truncation: bool = False
    ) -> dict:
        """Tokenize text(s)."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        all_ids = [self.encode(t) for t in texts]
        
        if truncation and max_length:
            all_ids = [ids[:max_length] for ids in all_ids]
        
        if padding:
            max_len = max(len(ids) for ids in all_ids)
            if max_length:
                max_len = min(max_len, max_length)
            
            attention_masks = []
            for i, ids in enumerate(all_ids):
                mask = [1] * len(ids)
                pad_len = max_len - len(ids)
                if pad_len > 0:
                    ids.extend([self.pad_token_id] * pad_len)
                    mask.extend([0] * pad_len)
                all_ids[i] = ids[:max_len]
                attention_masks.append(mask[:max_len])
        else:
            attention_masks = [[1] * len(ids) for ids in all_ids]
        
        result = {
            'input_ids': all_ids,
            'attention_mask': attention_masks
        }
        
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor(result['input_ids'], dtype=torch.long)
            result['attention_mask'] = torch.tensor(result['attention_mask'], dtype=torch.long)
        
        return result
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 8192,
        return_tensors: str = 'pt'
    ) -> dict:
        return self(
            texts,
            return_tensors=return_tensors,
            padding=True,
            max_length=max_length,
            truncation=True
        )


def text_to_bytes(text: str) -> torch.Tensor:
    """
    Quick utility to convert text to byte tensor.
    
    Args:
        text: Input string
    
    Returns:
        [1, len] tensor of byte values
    """
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode(text)
    return torch.tensor([tokens], dtype=torch.long)


def bytes_to_text(tensor: torch.Tensor) -> str:
    """
    Quick utility to convert byte tensor back to text.
    
    Args:
        tensor: [batch, seq] or [seq] tensor
    
    Returns:
        Decoded string
    """
    tokenizer = ByteTokenizer()
    if tensor.dim() == 2:
        tensor = tensor[0]
    return tokenizer.decode(tensor.tolist())
