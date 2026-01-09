# Copyright 2025 Pathway Technology, Inc.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from data_utils import NarrativeDocument, Sentence


@dataclass
class CoherenceScore:
    """Multi-dimensional coherence score for a sentence."""
    temporal_score: float  # How well it fits temporally
    causal_score: float  # How well causal relationships are maintained
    thematic_score: float  # Thematic consistency with backstory
    character_score: float  # Character consistency
    overall_score: float  # Weighted average
    
    relevance_to_backstory: float  # How relevant to backstory (0-1)
    is_violation: bool  # Whether a violation is detected
    violation_type: Optional[str]  # Type of violation if any
    explanation: str  # Human-readable explanation


@dataclass
class DevelopmentInfo:
    """Information about character/plot development."""
    entity_name: str
    development_type: str  # 'character_growth', 'plot_continuation', 'new_event'
    confidence: float
    description: str


class CoherenceScorer(nn.Module):
    """Multi-dimensional scoring for narrative coherence."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Temporal coherence scorer
        self.temporal_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Causal coherence scorer
        self.causal_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Thematic coherence scorer
        self.thematic_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Character consistency scorer
        self.character_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Violation detector
        self.violation_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        current_sentence_emb: torch.Tensor,
        backstory_context_emb: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute coherence scores.
        Args:
            current_sentence_emb: [d_model] embedding of current sentence
            backstory_context_emb: [d_model] relevant backstory context
        Returns:
            Dictionary of scores
        """
        # Concatenate embeddings
        combined = torch.cat([current_sentence_emb, backstory_context_emb], dim=0)  # [2*d_model]
        combined = combined.unsqueeze(0)  # [1, 2*d_model]
        
        # Compute individual scores
        temporal_score = self.temporal_scorer(combined).squeeze()
        causal_score = self.causal_scorer(combined).squeeze()
        thematic_score = self.thematic_scorer(combined).squeeze()
        character_score = self.character_scorer(combined).squeeze()
        
        # Violation probability
        violation_prob = self.violation_detector(combined).squeeze()
        
        return {
            'temporal': temporal_score,
            'causal': causal_score,
            'thematic': thematic_score,
            'character': character_score,
            'violation_prob': violation_prob
        }


class ViolationDetector:
    """Identifies contradictions and inconsistencies."""
    
    VIOLATION_THRESHOLD = 0.6
    
    @staticmethod
    def detect_violations(
        scores: Dict[str, torch.Tensor],
        current_sentence: Sentence,
        backstory_entities: Dict[str, Dict]
    ) -> Tuple[bool, Optional[str], str]:
        """
        Detect if there's a violation.
        Returns:
            (is_violation, violation_type, explanation)
        """
        # Check violation probability
        violation_prob = scores['violation_prob'].item()
        
        if violation_prob > ViolationDetector.VIOLATION_THRESHOLD:
            # Determine violation type based on which score is lowest
            temporal = scores['temporal'].item()
            causal = scores['causal'].item()
            thematic = scores['thematic'].item()
            character = scores['character'].item()
            
            min_score = min(temporal, causal, thematic, character)
            
            if min_score == temporal:
                return True, 'temporal_violation', f"Temporal inconsistency detected (score: {temporal:.2f})"
            elif min_score == causal:
                return True, 'causal_violation', f"Causal relationship violation (score: {causal:.2f})"
            elif min_score == thematic:
                return True, 'thematic_violation', f"Thematic inconsistency with backstory (score: {thematic:.2f})"
            else:
                return True, 'character_violation', f"Character inconsistency detected (score: {character:.2f})"
        
        return False, None, "No violations detected"


class DevelopmentTracker:
    """Tracks character and plot development."""
    
    DEVELOPMENT_THRESHOLD = 0.7
    
    def __init__(self):
        self.tracked_entities: Dict[str, List[int]] = {}  # entity -> sentence indices
    
    def track_development(
        self,
        sentence: Sentence,
        sentence_idx: int,
        scores: Dict[str, torch.Tensor],
        backstory_entities: Dict[str, Dict]
    ) -> List[DevelopmentInfo]:
        """
        Track development of entities in current sentence.
        Returns list of development information.
        """
        developments = []
        
        # Track entities in this sentence
        for entity in sentence.entities:
            # Initialize tracking if first occurrence
            if entity not in self.tracked_entities:
                self.tracked_entities[entity] = []
            
            self.tracked_entities[entity].append(sentence_idx)
            
            # Check if entity exists in backstory
            if entity in backstory_entities:
                # Existing character - check for development
                character_score = scores['character'].item()
                
                if character_score > self.DEVELOPMENT_THRESHOLD:
                    developments.append(DevelopmentInfo(
                        entity_name=entity,
                        development_type='character_growth',
                        confidence=character_score,
                        description=f"{entity} shows consistent development from backstory"
                    ))
            else:
                # New character
                developments.append(DevelopmentInfo(
                    entity_name=entity,
                    development_type='new_event',
                    confidence=0.8,
                    description=f"{entity} is a new character not in backstory"
                ))
        
        return developments


class SentenceAnalyzer(nn.Module):
    """Processes each sentence of current story and computes relevance to backstory."""
    
    def __init__(self, bdh_model, d_model: int, device: torch.device):
        super().__init__()
        
        self.bdh_model = bdh_model
        self.d_model = d_model
        self.device = device
        
        # Coherence scorer
        self.coherence_scorer = CoherenceScorer(d_model)
        
        # Violation detector
        self.violation_detector = ViolationDetector()
        
        # Development tracker
        self.development_tracker = DevelopmentTracker()
        
        # Context retrieval (simple attention-based)
        self.context_retrieval = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    
    def encode_sentence(self, sentence: Sentence) -> torch.Tensor:
        """Encode a single sentence using BDH."""
        tokens = torch.tensor(sentence.tokens[:512], dtype=torch.long, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            emb = self.bdh_model.embed(tokens)  # [1, seq_len, d_model]
        
        # Mean pool
        sent_emb = torch.mean(emb, dim=1).squeeze(0)  # [d_model]
        return sent_emb
    
    def retrieve_relevant_backstory(
        self,
        current_sentence_emb: torch.Tensor,
        backstory_sentence_embs: torch.Tensor,
        top_k: int = 5
    ) -> torch.Tensor:
        """
        Retrieve most relevant backstory sentences for current sentence.
        Args:
            current_sentence_emb: [d_model]
            backstory_sentence_embs: [num_backstory_sentences, d_model]
            top_k: Number of relevant sentences to retrieve
        Returns:
            [d_model] aggregated context from relevant backstory
        """
        # Compute similarity scores
        current_emb_norm = F.normalize(current_sentence_emb.unsqueeze(0), dim=-1)  # [1, d_model]
        backstory_emb_norm = F.normalize(backstory_sentence_embs, dim=-1)  # [N, d_model]
        
        similarities = torch.matmul(current_emb_norm, backstory_emb_norm.t()).squeeze(0)  # [N]
        
        # Get top-k most similar
        top_k = min(top_k, len(similarities))
        top_indices = torch.topk(similarities, k=top_k).indices
        
        # Retrieve and aggregate
        relevant_embs = backstory_sentence_embs[top_indices]  # [top_k, d_model]
        
        # Use attention to aggregate
        query = current_sentence_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        key_value = relevant_embs.unsqueeze(0)  # [1, top_k, d_model]
        
        context, _ = self.context_retrieval(query, key_value, key_value)
        context = context.squeeze(0).squeeze(0)  # [d_model]
        
        return context
    
    def analyze_sentence(
        self,
        sentence: Sentence,
        sentence_idx: int,
        backstory_sentence_embs: torch.Tensor,
        backstory_entities: Dict[str, Dict]
    ) -> CoherenceScore:
        """
        Analyze a single sentence from current story.
        Args:
            sentence: Current story sentence
            sentence_idx: Index of sentence
            backstory_sentence_embs: All backstory sentence embeddings
            backstory_entities: Entity information from backstory
        Returns:
            CoherenceScore object
        """
        # Encode current sentence
        current_emb = self.encode_sentence(sentence)
        
        # Retrieve relevant backstory context
        backstory_context = self.retrieve_relevant_backstory(
            current_emb,
            backstory_sentence_embs,
            top_k=5
        )
        
        # Compute coherence scores
        scores = self.coherence_scorer(current_emb, backstory_context)
        
        # Detect violations
        is_violation, violation_type, explanation = self.violation_detector.detect_violations(
            scores, sentence, backstory_entities
        )
        
        # Track development
        developments = self.development_tracker.track_development(
            sentence, sentence_idx, scores, backstory_entities
        )
        
        # Compute overall score (weighted average)
        temporal = scores['temporal'].item()
        causal = scores['causal'].item()
        thematic = scores['thematic'].item()
        character = scores['character'].item()
        
        overall = (temporal * 0.25 + causal * 0.25 + thematic * 0.3 + character * 0.2)
        
        # Relevance to backstory (inverse of violation probability)
        relevance = 1.0 - scores['violation_prob'].item()
        
        # Add development info to explanation
        if developments:
            dev_text = "; ".join([d.description for d in developments])
            explanation += f" | Developments: {dev_text}"
        
        return CoherenceScore(
            temporal_score=temporal,
            causal_score=causal,
            thematic_score=thematic,
            character_score=character,
            overall_score=overall,
            relevance_to_backstory=relevance,
            is_violation=is_violation,
            violation_type=violation_type,
            explanation=explanation
        )
    
    def analyze_document(
        self,
        current_story: NarrativeDocument,
        backstory_sentence_embs: torch.Tensor,
        backstory_entities: Dict[str, Dict]
    ) -> List[CoherenceScore]:
        """
        Analyze entire current story document.
        Returns list of coherence scores for each sentence.
        """
        results = []
        
        for idx, sentence in enumerate(current_story.sentences):
            score = self.analyze_sentence(
                sentence, idx, backstory_sentence_embs, backstory_entities
            )
            results.append(score)
        
        return results
