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
    
    def forward(self, surprise_pos: float, surprise_neg: float, 
                is_contradict: bool) -> torch.Tensor:
        """
        Args:
            surprise_pos: Surprise of original statement
            surprise_neg: Surprise of negated statement
            is_contradict: True if label is "contradict"
        
        Returns:
            Loss tensor
        """
        E_pos = torch.tensor(surprise_pos, dtype=torch.float32)
        E_neg = torch.tensor(surprise_neg, dtype=torch.float32)
        
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
    
    def __init__(self, bdh_model, device):
        self.bdh = bdh_model
        self.device = device
        
        # Negation patterns (rule-based, no LLM)
        self.negation_patterns = [
            (r"^(.+) is (.+)$", r"\1 is NOT \2"),
            (r"^(.+) was (.+)$", r"\1 was NOT \2"),
            (r"^(.+) has (.+)$", r"\1 has NOT \2"),
            (r"^(.+) had (.+)$", r"\1 had NOT \2"),
            (r"^(.+) did (.+)$", r"\1 did NOT \2"),
            (r"^(.+) does (.+)$", r"\1 does NOT \2"),
            (r"^(.+) can (.+)$", r"\1 can NOT \2"),
            (r"^(.+) will (.+)$", r"\1 will NOT \2"),
        ]
    
    def negate(self, statement: str) -> str:
        """
        Generate negated variant of statement.
        Rule-based, no external LLM required.
        """
        statement = statement.strip()
        
        # Try each pattern
        for pattern, replacement in self.negation_patterns:
            if re.match(pattern, statement, re.IGNORECASE):
                return re.sub(pattern, replacement, statement, flags=re.IGNORECASE)
        
        # Fallback: prepend negation phrase
        return f"It is NOT true that {statement}"
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Convert text to byte tokens."""
        tokens = torch.tensor([[ord(c) % 256 for c in text]], dtype=torch.long, device=self.device)
        return tokens
    
    def compute_surprise(self, text: str, world_state: torch.Tensor, 
                          fact_time: float = 0.0, query_time: float = 1.0,
                          temporal_beta: float = 0.01) -> float:
        """
        Measure how much the statement "surprises" the world state.
        
        High surprise = statement conflicts with stored facts.
        
        Bugfixes Applied:
        - Fix #1: detach().clone() for safe state copying
        - Fix #2: Layer-weighted Δ (later layers weighted more)
        - Fix #5: Temporal decay affects decision
        - Fix #6: log1p stabilization
        """
        # Safe state cloning - no gradient leakage
        state_before = world_state.detach().clone().to(self.device)
        
        # Reset before setting to prevent silent drift
        self.bdh.reset_state()
        self.bdh.set_state(state_before)
        
        # Encode statement
        tokens = self.encode_text(text)
        
        with torch.no_grad():
            self.bdh(tokens, use_state=True)
        
        # Get state after
        state_after = self.bdh.get_state()
        
        # Bugfix #2: Layer-weighted Δ (later layers encode higher-level state)
        if state_after.dim() >= 2 and state_after.shape[0] > 1:
            n_layers = state_after.shape[0]
            layer_weights = torch.linspace(0.5, 1.5, n_layers, device=self.device)
            
            # Compute weighted sum of per-layer deltas
            delta = 0.0
            for i in range(n_layers):
                layer_delta = torch.norm(state_after[i] - state_before[i], p=2).item()
                delta += layer_weights[i].item() * layer_delta
        else:
            # Fallback for single-layer or flat state
            delta = torch.norm(state_after - state_before, p=2).item()
        
        # Temporal decay affects surprise
        temporal_decay = math.exp(-temporal_beta * (query_time - fact_time))
        delta = delta * temporal_decay
        
        # Stabilize via log1p (prevents explosion)
        surprise = math.log1p(delta)
        
        # Clamp for additional safety
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
