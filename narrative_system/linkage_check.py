import torch
import torch.nn.functional as F

class NarrativeLinkageChecker:
    """
    Research-principled Linkage detection.
    Based on the assumption that a 'linked' backstory reduces the entropy of the current story.
    Score = log P(Statement | Backstory) - log P(Statement | Default)
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def compute_log_likelihood(self, text, use_backstory=True):
        """
        Calculates the average log-probability of a text.
        """
        # 1. Encode
        text = str(text).strip()
        tokens_list = self.tokenizer.encode(text) if text else [0]
        
        # Handle too short sequences for causal shift
        if len(tokens_list) < 2:
             tokens_list = tokens_list + [0]
             
        tokens = torch.tensor([tokens_list], device=self.device)
        targets = tokens.clone()
        
        # 2. Forward pass
        # If not using backstory, we reset the model state before the call.
        if not use_backstory:
            self.model.reset_state()
            
        with torch.no_grad():
            logits, _, _, linkage_logit = self.model(tokens, use_state=True)
            
        # 3. Calculate mean log-probs
        log_probs = F.log_softmax(logits, dim=-1)
        targets = targets[:, 1:].unsqueeze(-1)
        log_probs = log_probs[:, :-1, :]
        
        gathered = torch.gather(log_probs, -1, targets).squeeze(-1)
        return gathered.mean().item(), linkage_logit

    def check_linkage(self, statement, backstory_state=None):
        """
        Decision function using Hybrid PMI + Semantic logic.
        """
        # P(Statement | Backstory)
        if backstory_state is not None:
             for i, block in enumerate(self.model.transformer.h):
                 if backstory_state[i].shape != block.attn.state.shape:
                      continue
                 block.attn.state.copy_(backstory_state[i].to(self.device))
        
        lp_conditional, linkage_logit = self.compute_log_likelihood(statement, use_backstory=True)
        
        # P(Statement | Empty)
        lp_marginal, _ = self.compute_log_likelihood(statement, use_backstory=False)
        
        # PMI = Conditional - Marginal
        pmi = lp_conditional - lp_marginal
        
        # Semantic Prob from Classifier
        semantic_prob = torch.sigmoid(linkage_logit).item()
        
        # Fusion Decision:
        # We trust the semantic classifier but use PMI as a sanity check.
        # A statement is consistent if it's semantically likely AND logically plausible.
        is_consistent = (semantic_prob > 0.5) and (pmi > -0.5) 
        
        label = "consistent" if is_consistent else "contradict"
        
        # Return a score that balances probabilistic and semantic confidence
        confidence = 0.5 * semantic_prob + 0.5 * (1.0 if pmi > 0 else 0.0)
        
        return label, confidence
