# Copyright 2025 Pathway Technology, Inc.

import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import torch
import numpy as np


@dataclass
class NarrativeEntity:
    """Represents an entity in the narrative (character, location, event)."""
    name: str
    entity_type: str  # 'character', 'location', 'event', 'theme'
    mentions: List[int]  # sentence indices where entity appears
    attributes: Dict[str, str]  # extracted attributes


@dataclass
class Sentence:
    """Represents a single sentence with metadata."""
    text: str
    index: int
    tokens: List[int]  # byte-level tokens
    entities: List[str]  # entity names mentioned
    temporal_marker: str  # 'past', 'present', 'future', 'unknown'


@dataclass
class NarrativeDocument:
    """Structured representation of a narrative text."""
    sentences: List[Sentence]
    entities: Dict[str, NarrativeEntity]
    raw_text: str
    

class TextLoader:
    """Load and preprocess text files for narrative analysis."""
    
    @staticmethod
    def load_text(file_path: str) -> str:
        """Load text from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize line endings
        text = text.replace('\r\n', '\n')
        return text.strip()


class SentenceTokenizer:
    """Tokenize text into sentences using rule-based approach."""
    
    # Common abbreviations to avoid false sentence breaks
    ABBREVIATIONS = {
        'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.',
        'etc.', 'vs.', 'i.e.', 'e.g.', 'a.m.', 'p.m.'
    }
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence boundary detection
        # (Can be replaced with NLTK/spaCy for production)
        sentences = []
        
        # Split on sentence-ending punctuation followed by whitespace and capital letter
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        splits = re.split(pattern, text)
        
        current_sentence = ""
        for split in splits:
            current_sentence += split
            # Check if ends with sentence-ending punctuation
            if re.search(r'[.!?]$', split.strip()):
                # Check if it's not an abbreviation
                last_word = split.strip().split()[-1].lower()
                if last_word not in SentenceTokenizer.ABBREVIATIONS:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Add remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences


class EntityExtractor:
    """Extract entities from text using rule-based patterns."""
    
    # Patterns for entity extraction
    CHARACTER_PATTERNS = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized names
    ]
    
    TEMPORAL_MARKERS = {
        'past': ['was', 'were', 'had', 'did', 'ago', 'yesterday', 'before', 'previously', 'earlier'],
        'present': ['is', 'are', 'am', 'now', 'currently', 'today'],
        'future': ['will', 'shall', 'going to', 'tomorrow', 'later', 'soon', 'next'],
    }
    
    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """Extract potential entity names from text."""
        entities = set()
        
        # Extract capitalized names
        for pattern in EntityExtractor.CHARACTER_PATTERNS:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        # Filter out common words and short names
        entities = {e for e in entities if len(e) > 2 and e.lower() not in ['the', 'and', 'but']}
        
        return list(entities)
    
    @staticmethod
    def detect_temporal_marker(sentence: str) -> str:
        """Detect temporal marker in sentence."""
        sentence_lower = sentence.lower()
        
        for tense, markers in EntityExtractor.TEMPORAL_MARKERS.items():
            if any(marker in sentence_lower for marker in markers):
                return tense
        
        return 'unknown'


class BPETokenizer:
    """Convert text to BPE tokens compatible with BDH and pretrained decoders."""
    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to BPE tokens."""
        return self.enc.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode BPE tokens to text."""
        return self.enc.decode(tokens)
    
    def encode_batch(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """Encode batch of texts with padding."""
        encoded = []
        for text in texts:
            tokens = self.encode(text)
            # Truncate or pad
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))
            encoded.append(tokens)
        
        return torch.tensor(encoded, dtype=torch.long)


class NarrativeDocumentBuilder:
    """Build structured narrative document from raw text."""
    
    def __init__(self):
        self.sentence_tokenizer = SentenceTokenizer()
        self.entity_extractor = EntityExtractor()
        self.tokenizer = BPETokenizer()
    
    def build(self, text: str) -> NarrativeDocument:
        """Build narrative document from text."""
        # Clean text
        text = TextLoader.clean_text(text)
        
        # Tokenize into sentences
        sentence_texts = self.sentence_tokenizer.tokenize(text)
        
        # Extract entities across all sentences
        all_entities: Dict[str, NarrativeEntity] = {}
        
        # Build sentence objects
        sentences = []
        for idx, sent_text in enumerate(sentence_texts):
            # Extract entities from sentence
            sent_entities = self.entity_extractor.extract_entities(sent_text)
            
            # Update global entity tracking
            for entity_name in sent_entities:
                if entity_name not in all_entities:
                    all_entities[entity_name] = NarrativeEntity(
                        name=entity_name,
                        entity_type='character',  # Default, can be refined
                        mentions=[],
                        attributes={}
                    )
                all_entities[entity_name].mentions.append(idx)
            
            # Detect temporal marker
            temporal_marker = self.entity_extractor.detect_temporal_marker(sent_text)
            
            # Create sentence object
            sentence = Sentence(
                text=sent_text,
                index=idx,
                tokens=self.tokenizer.encode(sent_text),
                entities=sent_entities,
                temporal_marker=temporal_marker
            )
            sentences.append(sentence)
        
        return NarrativeDocument(
            sentences=sentences,
            entities=all_entities,
            raw_text=text
        )


class ConstraintExtractor:
    """Extract constraints from backstory."""
    
    @staticmethod
    def extract_character_constraints(document: NarrativeDocument) -> Dict[str, Dict]:
        """Extract character-related constraints."""
        constraints = {}
        
        for entity_name, entity in document.entities.items():
            if entity.entity_type == 'character':
                # Collect sentences mentioning this character
                character_sentences = [
                    document.sentences[idx].text 
                    for idx in entity.mentions
                ]
                
                constraints[entity_name] = {
                    'mentioned_in': entity.mentions,
                    'sentence_count': len(entity.mentions),
                    'contexts': character_sentences[:5]  # First 5 mentions
                }
        
        return constraints
    
    @staticmethod
    def extract_temporal_constraints(document: NarrativeDocument) -> List[Tuple[int, int, str]]:
        """Extract temporal ordering constraints (sentence_idx1, sentence_idx2, relation)."""
        constraints = []
        
        # Simple temporal ordering based on text position
        for i in range(len(document.sentences) - 1):
            # Consecutive sentences have temporal ordering
            constraints.append((i, i + 1, 'before'))
        
        return constraints
    
    @staticmethod
    def extract_causal_patterns(document: NarrativeDocument) -> List[Tuple[int, int]]:
        """Extract potential causal relationships between sentences."""
        causal_keywords = ['because', 'therefore', 'thus', 'so', 'hence', 'as a result', 'consequently']
        causal_pairs = []
        
        for i, sentence in enumerate(document.sentences):
            text_lower = sentence.text.lower()
            # If sentence contains causal keyword, it might be caused by previous sentence
            if any(keyword in text_lower for keyword in causal_keywords):
                if i > 0:
                    causal_pairs.append((i - 1, i))  # Previous sentence causes this one
        
        return causal_pairs


def pad_sequence_batch(sequences: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
    """Pad sequences to same length."""
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padding = torch.full((max_len - len(seq),), padding_value, dtype=seq.dtype)
            padded.append(torch.cat([seq, padding]))
        else:
            padded.append(seq)
    return torch.stack(padded)
