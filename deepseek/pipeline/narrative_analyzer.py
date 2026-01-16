import torch
from typing import List, Dict, Any
from .bdh_state_manager import BDHStateManager

class NarrativeAnalyzer:
    def __init__(self, state_manager: BDHStateManager, classifier: torch.nn.Module):
        self.state_manager = state_manager
        self.classifier = classifier

    def analyze_consistency(self, char_name: str, backstory_embeddings: torch.Tensor) -> Dict[str, Any]:
        """
        Main analysis flow:
        1. Retrieve character's persistent states from BDHStateManager.
        2. Run ConsistencyClassifier with attention evidence weighting.
        3. Extract evidence rationale.
        """
        # Get character narrative history
        narrative_history = self.state_manager.character_states.get(char_name, [])
        if not narrative_history:
            return {"prediction": 1, "confidence": 0.5, "rationale": "No context found for character."}
        
        # Stack embeddings: [1, S_n, D]
        narrative_tensor = torch.stack(narrative_history).unsqueeze(0)
        
        self.classifier.eval()
        with torch.no_grad():
            logits, weights = self.classifier(narrative_tensor, backstory_embeddings.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, prediction].item()
            
        # Rationale Generation (simple version for now)
        rationale = self.generate_rationale(prediction, confidence, weights)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "rationale": rationale
        }

    def generate_rationale(self, prediction: int, confidence: float, weights: torch.Tensor) -> str:
        # Placeholder for complex rationale generation
        status = "consistent" if prediction == 1 else "contradictory"
        return f"The backstory is {status} with the narrative context (Confidence: {confidence:.2f})."