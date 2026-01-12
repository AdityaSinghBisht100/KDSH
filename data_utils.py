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


