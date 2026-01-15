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
            logits, _ = self.model(tokens, use_state=True)
            
        # 3. Calculate mean log-softman for targets
        # logits shape: [1, T, Vocab]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather the log-probs for the target tokens
        # Shifted by 1 for causal prediction
        targets = targets[:, 1:].unsqueeze(-1)
        log_probs = log_probs[:, :-1, :]
        
        gathered = torch.gather(log_probs, -1, targets).squeeze(-1)
        return gathered.mean().item()

    def check_linkage(self, statement, backstory_state=None):
        """
        Main decision function.
        Returns: ("linked"|"unlinked", confidence_score)
        """
        # P(Statement | Backstory)
        if backstory_state is not None:
             # Load state into model before checking
             for i, block in enumerate(self.model.transformer.h):
                 if backstory_state[i].shape != block.attn.state.shape:
                      continue # Skip incompatible states or handle error
                 block.attn.state.copy_(backstory_state[i].to(self.device))
        
        lp_conditional = self.compute_log_likelihood(statement, use_backstory=True)
        
        # P(Statement | Empty)
        lp_marginal = self.compute_log_likelihood(statement, use_backstory=False)
        
        # PMI = Conditional - Marginal
        pmi = lp_conditional - lp_marginal
        
        # Decision: If the backstory helps predict the statement (Higher PMI), it is linked.
        # Threshold 0.0 means any improvement in probability counts.
        label = "consistent" if pmi > 0 else "contradict"
        
        return label, pmi
