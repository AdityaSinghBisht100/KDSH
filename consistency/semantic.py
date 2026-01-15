"""
Semantic Consistency Checker

Uses cosine similarity between statement and backstory embeddings
to determine consistency.
"""
import torch
from typing import Tuple, Optional

class SemanticConsistencyChecker:
    """
    Checks narrative consistency using semantic similarity.
    """
    
    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: Similarity threshold for consistency.
                      Higher = stricter (more likely to say contradict)
        """
        self.threshold = threshold
    
    def check(
        self,
        statement_embedding: torch.Tensor,
        backstory_embedding: torch.Tensor
    ) -> Tuple[str, float]:
        """
        Check if statement is consistent with backstory.
        
        Args:
            statement_embedding: [embedding_dim] tensor
            backstory_embedding: [embedding_dim] tensor
            
        Returns:
            (label, similarity_score)
        """
        # Ensure correct shape
        if statement_embedding.dim() == 1:
            statement_embedding = statement_embedding.unsqueeze(0)
        if backstory_embedding.dim() == 1:
            backstory_embedding = backstory_embedding.unsqueeze(0)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            statement_embedding,
            backstory_embedding
        ).item()
        
        # Apply threshold
        # Higher similarity = more likely consistent
        # But contradictions often have negative or low similarity
        if similarity > self.threshold:
            label = "consistent"
        else:
            label = "contradict"
        
        return label, similarity
    
    def check_with_context(
        self,
        statement_embedding: torch.Tensor,
        backstory_embedding: torch.Tensor,
        global_context: Optional[torch.Tensor] = None,
        context_weight: float = 0.2
    ) -> Tuple[str, float]:
        """
        Check consistency with optional global context.
        """
        if global_context is not None:
            # Merge backstory with global context
            merged = (1 - context_weight) * backstory_embedding + context_weight * global_context
        else:
            merged = backstory_embedding
        
        return self.check(statement_embedding, merged)
