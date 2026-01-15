class NarrativeLinkageChecker:
    """
    Semantic-Principled Linkage detection.
    Compares the 'Soul Vector' of a Statement against the Backstory state.
    """
    def __init__(self, model, encoder, device):
        self.model = model
        self.encoder = encoder
        self.device = device
        
    def get_semantic_representation(self, text, state=None):
        """
        Refines a text's semantic vector using the provided backstory state.
        """
        text = str(text).strip()
        if not text:
             return torch.zeros((1, 256), device=self.device), torch.tensor([0.0], device=self.device)
             
        # 1. Base Semantic Embedding [1, 384]
        with torch.no_grad():
            vec_np = self.encoder.encode([text], convert_to_numpy=True)
            vec = torch.from_numpy(vec_np).to(self.device).unsqueeze(1) # [1, 1, 384]
            
        # 2. Refinement through BDH Memory
        if state is not None:
             self.model.set_state(state)
        else:
             self.model.reset_state()
             
        with torch.no_grad():
            # refined_vec: [1, 256], logit: [1, 1]
            refined_vec, linkage_logit, _ = self.model(vec, use_state=True)
            
        return refined_vec, linkage_logit

    def check_linkage(self, statement, backstory_state=None):
        """
        Decision function using Hybrid Semantic Logic.
        """
        # 1. Statement logic IN CONTEXT
        vec_ctx, logit_ctx = self.get_semantic_representation(statement, backstory_state)
        
        # 2. Statement logic IN ISOLATION (Baseline)
        vec_base, _ = self.get_semantic_representation(statement, None)
        
        # 3. Decision Signals
        # Signal A: Semantic Classifier (Trained on supervised labels)
        semantic_prob = torch.sigmoid(logit_ctx).item()
        
        # Signal B: State Shift (How much did the context change the meaning?)
        # A consistent statement should have a high cosine similarity with its baseline
        # but also be "expected" by the memory.
        similarity = F.cosine_similarity(vec_ctx, vec_base).item()
        
        # Fusion:
        # A statement is consistent if the classifier says so AND it doesn't radically 
        # shift the semantic meaning away from its grounded baseline in a "confused" way.
        is_consistent = (semantic_prob > 0.5) and (similarity > 0.7)
        
        label = "consistent" if is_consistent else "contradict"
        
        # Confidence blends the classifier certainty and the semantic stability
        confidence = 0.6 * semantic_prob + 0.4 * similarity
        
        return label, confidence
