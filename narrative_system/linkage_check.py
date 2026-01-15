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
        tokens = torch.tensor([self.tokenizer.encode(text)], device=self.device)
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
        Returns: ("consistent"|"contradict", pmi_score)
        
        Uses PMI: log P(statement|backstory) - log P(statement|empty)
        - Positive PMI: backstory helps predict statement → consistent
        - Negative PMI: backstory hurts prediction → contradict
        """
        # CRITICAL FIX: Save current state before any modifications
        saved_state = [block.attn.state.clone() for block in self.model.transformer.h]
        
        # Step 1: Load backstory state for conditional computation
        states_loaded = 0
        if backstory_state is not None:
            for i, block in enumerate(self.model.transformer.h):
                if i < len(backstory_state) and backstory_state[i].shape == block.attn.state.shape:
                    block.attn.state.copy_(backstory_state[i].to(self.device))
                    states_loaded += 1
            
            if states_loaded == 0:
                print("❌ No backstory states loaded due to shape mismatches!")
        
        # Step 2: Compute conditional log-likelihood P(statement | backstory)
        lp_conditional = self.compute_log_likelihood(statement, use_backstory=True)
        
        # Step 3: Compute marginal log-likelihood P(statement | empty)
        # Reset clears state, so we get baseline probability
        lp_marginal = self.compute_log_likelihood(statement, use_backstory=False)
        
        # Step 4: Restore original state (CRITICAL - was missing!)
        for i, block in enumerate(self.model.transformer.h):
            block.attn.state.copy_(saved_state[i])
        
        # PMI = Conditional - Marginal
        pmi = lp_conditional - lp_marginal
        
        # Decision logic:
        # If backstory helps predict statement (higher probability) → consistent
        # If backstory hurts prediction (lower probability) → contradict
        label = "consistent" if pmi > 0 else "contradict"
        
        return label, pmi
