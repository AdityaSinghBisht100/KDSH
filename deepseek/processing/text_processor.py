import re
from typing import List, Tuple

class TextProcessor:
    def __init__(self, config):
        self.config = config

    def extract_character_mentions(self, text: str, char_name: str) -> List[str]:
        """
        Extracts sentences or paragraphs containing the character name.
        """
        # Split into paragraphs for context
        paragraphs = text.split('\n\n')
        mentions = []
        for p in paragraphs:
            if char_name.lower() in p.lower():
                mentions.append(p.strip())
        return mentions

    def chunk_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """
        Chunks text into manageable sizes for BDH processing.
        """
        # Simple overlap-free chunking for now
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
