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
        
    def encode_text(self, text: str) -> torch.Tensor:
        """Convert text to BPE tokens using tiktoken."""
        # Standardize: always use tiktoken.
        ids = self.tokenizer.encode(str(text))
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    
    def compute_energy(self, text: str, world_state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Average Per-Token Cross Entropy (Energy) 
        of the text given a specific world state.
        
        This uses the 'Actual BDH Thinking'—the model's internal probability.
        """
        # Load state
        self.bdh.reset_state()
        if world_state is not None:
             self.bdh.set_state(world_state.detach().clone().to(self.device))
        
        # Prepare inputs
        tokens = self.encode_text(text)
        if tokens.size(1) < 2:
             return torch.tensor(10.0, device=self.device) # High energy for tiny/malformed text
             
        # Targets are just shifted tokens
        targets = tokens.clone()
        
        # Shift inputs/targets for auto-regressive loss
        # input: [B, T-1], target: [B, T-1] (shifted by 1)
        inputs = tokens[:, :-1]
        targets = targets[:, 1:]
        
        # Forward Pass
        # We use use_state=True to ensure the world state matrix is involved in the calculation
        logits, loss = self.bdh(inputs, targets=targets, use_state=True)
        
        # Loss is already averaged over T by F.cross_entropy in bdh.py
        return loss

    def predict(self, statement: str, world_state: torch.Tensor) -> Tuple[str, float]:
        """
        Uses Differential Energy to determine consistency.
        
        Differential Energy = E(S | World) - E(S | Reset)
        
        - Negative Delta: World State REDUCED the surprise of the statement. (Consistent)
        - Positive Delta: World State INCREASED the surprise. (Contradict)
        """
        # 1. Compute Energy with the narrative context
        energy_context = self.compute_energy(statement, world_state)
        
        # 2. Compute Energy without any context (reset state)
        energy_base = self.compute_energy(statement, None)
        
        # 3. Decision Logic
        # We use a small buffer (0.01) to avoid noise-driven contradictions
        delta = (energy_context - energy_base).item()
        
        if delta > 0.01:
            # The world state was confused by this statement
            return "contradict", 1.0 + abs(delta)
        else:
            # The world state explained this statement
            return "consistent", 1.0 + abs(delta)
    
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
    
    def predict_with_details(self, statement: str, world_state: torch.Tensor) -> Dict:
        """
        Detailed prediction with Energy Deltas.
        """
        energy_context = self.compute_energy(statement, world_state).item()
        energy_base = self.compute_energy(statement, None).item()
        delta = energy_context - energy_base
        
        prediction = "contradict" if delta > 0.01 else "consistent"
        
        return {
            "statement": statement,
            "energy_with_context": energy_context,
            "energy_without_context": energy_base,
            "energy_delta": delta,
            "prediction": prediction
        }
