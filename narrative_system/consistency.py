import torch
import torch.nn as nn
import re
import math
from typing import Tuple, Dict, List

# Maximum surprise value for clamping (Review Fix #6)
SURPRISE_MAX = 100.0

class ContrastiveEnergyLoss(nn.Module):
    """
    Contrastive Energy Loss for supervision.
    
    Aligns surprise with labels via margin loss.
    Contradict samples should have higher E_pos than E_neg.
    """
    def __init__(self, margin: float = 0.3):
        """Margin increased to 0.3 for stronger separation."""
        super().__init__()
        self.margin = margin
    
    def forward(self, surprise_pos, surprise_neg, 
                is_contradict: bool) -> torch.Tensor:
        """
        Args:
            surprise_pos: Surprise of original statement (tensor or float)
            surprise_neg: Surprise of negated statement (tensor or float)
            is_contradict: True if label is "contradict"
        
        Returns:
            Loss tensor
        """
        # Preserve gradient chain if inputs are already tensors
        if isinstance(surprise_pos, torch.Tensor):
            E_pos = surprise_pos
        else:
            E_pos = torch.tensor(surprise_pos, dtype=torch.float32, requires_grad=True)
            
        if isinstance(surprise_neg, torch.Tensor):
            E_neg = surprise_neg
        else:
            E_neg = torch.tensor(surprise_neg, dtype=torch.float32, requires_grad=True)
        
        if is_contradict:
            # Contradiction: E_pos should be HIGH (resisted by world)
            # E_neg should be LOW (accepted by world)
            loss = torch.clamp(self.margin + E_neg - E_pos, min=0)
        else:
            # Consistent: E_pos should be LOW (accepted)
            # E_neg should be HIGH (resisted)
            loss = torch.clamp(self.margin + E_pos - E_neg, min=0)
        
        return loss


class CounterfactualChecker:
    """
    Counterfactual Consistency Checking.
    
    Detects LOGICAL contradictions by comparing:
    - How much the statement "surprises" the world state
    - How much its negation "surprises" the world state
    
    If statement causes MORE surprise → CONTRADICT
    If negation causes MORE surprise → CONSISTENT
    """
    
    def __init__(self, bdh_model, tokenizer, device):
        self.bdh = bdh_model
        self.tokenizer = tokenizer
        self.device = device
        
        # Enhanced Negation patterns (Ordered by specificity)
        # Group 1: Remove existing negation (Toggle)
        self.negation_removals = [
            (r"\b(is|was|are|were|has|have|had|did|do|does|can|could|should|would|will|must|might) not\b", r"\1"),
            (r"\b(isn|wasn|aren|weren|hasn|haven|hadn|didn|don|doesn|couldn|shouldn|wouldn|mustn)[\u2019']t\b", r"\1"),
            (r"\bcan[\u2019']t\b", "can"),
            (r"\bwon[\u2019']t\b", "will"),
            (r"\bnever\b", "always"),
            (r"\bno\b", "some"), # aggressive but often valid
        ]

        # Group 2: Add negation to positive verbs
        self.negation_additions = [
            (r"\b(is|was|are|were|has|have|had|did|do|does|can|could|should|would|will|must|might)\b", r"\1 not"),
            (r"\b(Is|Was|Are|Were|Has|Have|Had|Did|Do|Does|Can|Could|Should|Would|Will|Must|Might)\b", r"\1 not"), # Capitalized start
        ]

        # Group 3: Morphological handling (simple heuristics for past tense)
        # "He walked" -> "He did not walk" is hard without lemmas.
        # We stick to the robust fallback for complex verbs.

    def negate(self, statement: str) -> str:
        """
        Generate negated variant of statement.
        """
        # Robustness fix: Ensure statement is a string and handle NaN/Empty
        statement = str(statement).strip()
        if not statement or statement == 'nan':
            return "Nothing happened."
        
        # 1. Try to REMOVE negation first (Double Negation Logic)
        for pattern, replacement in self.negation_removals:
            if re.search(pattern, statement, re.IGNORECASE):
                return re.sub(pattern, replacement, statement, count=1, flags=re.IGNORECASE)
        
        # 2. Try to ADD negation to auxiliary verbs
        for pattern, replacement in self.negation_additions:
            # We use search to find the *first* verb match to negate main clause
            if re.search(pattern, statement): # Case sensitive for the 'Capitalized' check logic
                return re.sub(pattern, replacement, statement, count=1)
        
        # 3. Fallback: Distinctive Prefix
        # "It is false that" acts as a strong logical operator
        return f"It is false that {statement}"
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Convert text to BPE tokens."""
        ids = self.tokenizer.encode(text)
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        return tokens
    
    def compute_surprise(self, text: str, world_state: torch.Tensor, 
                          fact_time: float = 0.0, query_time: float = 1.0,
                          temporal_beta: float = 0.01, training: bool = False) -> float:
        """
        Measure how much the statement "surprises" the world state.
        
        High surprise = statement conflicts with stored facts.
        """
        # Safe state cloning - no gradient leakage
        state_before = world_state.detach().clone().to(self.device)
        
        # Reset before setting to prevent silent drift
        self.bdh.reset_state()
        self.bdh.set_state(state_before)
        
        # Encode statement
        tokens = self.encode_text(text)
        
        if training:
            # Functional Forward Pass for Energy Training
            # use_state=True: Run recurrence (so state evolves)
            # return_new_state=True: Return evolved state, do NOT update persistent self.state
            _, state_after = self.bdh(tokens, use_state=True, return_new_state=True)
        else:
            with torch.no_grad():
                self.bdh(tokens, use_state=True)
                # For inference, state is updated in-place, so we fetch it
                state_after = self.bdh.get_state()
        
        # Bugfix #2: Layer-weighted Δ (later layers encode higher-level state)
        if state_after.dim() >= 2 and state_after.shape[0] > 1:
            n_layers = state_after.shape[0]
            layer_weights = torch.linspace(0.5, 1.5, n_layers, device=self.device)
            
            # Compute weighted sum of per-layer deltas
            delta = torch.tensor(0.0, device=self.device)
            for i in range(n_layers):
                layer_delta = torch.norm(state_after[i] - state_before[i], p=2)
                delta += layer_weights[i] * layer_delta
        else:
            # Fallback for single-layer or flat state
            delta = torch.norm(state_after - state_before, p=2)
            
        if not training:
            delta = delta.item()
        
        # Temporal decay affects surprise
        temporal_decay = math.exp(-temporal_beta * (query_time - fact_time))
        delta = delta * temporal_decay
        
        # Stabilize via log1p (prevents explosion)
        if training:
             surprise = torch.log1p(delta)
        else:
             surprise = math.log1p(delta)
        
        # Clamp for additional safety
        if not training:
             surprise = min(surprise, math.log1p(SURPRISE_MAX))
        
        return surprise
    
    def predict(self, statement: str, world_state: torch.Tensor) -> Tuple[str, float]:
        """
        Counterfactual consistency prediction.
        
        Returns:
            (prediction, confidence)
            prediction: "consistent" or "contradict"
            confidence: ratio indicating strength of prediction
        """
        # Generate negation
        negated = self.negate(statement)
        
        # Compute surprise for both
        surprise_S = self.compute_surprise(statement, world_state)
        surprise_negS = self.compute_surprise(negated, world_state)
        
        # Avoid division by zero
        epsilon = 1e-8
        conflict_ratio = surprise_S / (surprise_negS + epsilon)
        
        # Decision logic
        if conflict_ratio > 1.0:
            # Statement caused MORE surprise than its negation
            # → World state conflicts with statement
            return "contradict", conflict_ratio
        else:
            # Negation caused MORE surprise
            # → World state aligns with statement  
            return "consistent", 1.0 / (conflict_ratio + epsilon)
    
    def predict_with_details(self, statement: str, world_state: torch.Tensor) -> Dict:
        """
        Detailed prediction with debug info.
        """
        negated = self.negate(statement)
        surprise_S = self.compute_surprise(statement, world_state)
        surprise_negS = self.compute_surprise(negated, world_state)
        
        epsilon = 1e-8
        conflict_ratio = surprise_S / (surprise_negS + epsilon)
        
        prediction = "contradict" if conflict_ratio > 1.0 else "consistent"
        
        return {
            "statement": statement,
            "negated": negated,
            "surprise_statement": surprise_S,
            "surprise_negation": surprise_negS,
            "conflict_ratio": conflict_ratio,
            "prediction": prediction
        }
