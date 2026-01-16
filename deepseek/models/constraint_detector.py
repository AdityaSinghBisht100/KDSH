# models/constraint_detector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class LearnedConstraintDetector(nn.Module):
    """Learns constraint types from data"""
    
    def __init__(self, input_dim=768, num_types=10):
        super().__init__()
        self.input_dim = input_dim
        
        self.constraint_classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_types)
        )
        
        self.polarity_detector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
        self.importance_scorer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.constraint_names = [
            "temporal", "causal", "character_trait", "capability",
            "obligation", "belief", "desire", "prohibition",
            "event_sequence", "general"
        ]
        
    def forward(self, encoding):
        constraint_logits = self.constraint_classifier(encoding)
        constraint_probs = F.softmax(constraint_logits, dim=-1)
        constraint_type = torch.argmax(constraint_probs, dim=-1)
        
        polarity_logits = self.polarity_detector(encoding)
        polarity_probs = F.softmax(polarity_logits, dim=-1)
        polarity_idx = torch.argmax(polarity_probs, dim=-1)
        polarity_map = {0: 1, 1: -1, 2: 0}
        polarity = polarity_map[polarity_idx.item()]
        
        importance = self.importance_scorer(encoding)
        
        return {
            'constraint_type': self.constraint_names[constraint_type.item()],
            'constraint_probs': constraint_probs,
            'polarity': polarity,
            'importance': importance.item(),
            'encoding': encoding
        }
    
    def extract_from_text(self, text, character=None):
        """Extract constraints from text"""
        sentences = re.split(r'[.!?]+', text)
        constraints = []
        
        for sentence in sentences:
            if character and not re.search(rf'\b{re.escape(character)}\b', sentence, re.IGNORECASE):
                continue
                
            # This would normally use BDH encoding
            # For now, return placeholder
            constraints.append({
                'sentence': sentence.strip(),
                'type': 'general',
                'polarity': 0,
                'importance': 0.5
            })
        
        return constraints