"""
Integration Test for SBERT and GPT-2 Decoder

Tests that all components work together without dimension conflicts.
"""
import torch

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    from bdh import (
        BDHConfig, BDH_GPU, SBERTEncoder, RationaleDecoder,
        RationaleDecoderWithBDH
    )
    print("✅ All imports successful")
    return True

def test_config():
    """Test config with SBERT parameters."""
    print("\nTesting config...")
    from bdh import BDHConfig
    
    config = BDHConfig(
        n_neurons=512,
        n_layers=2,
        n_heads=8,
        use_sbert=True,
        sbert_dim=384,
        gpt2_model='gpt2'
    )
    
    print(f"  n_neurons: {config.n_neurons}")
    print(f"  sbert_dim: {config.sbert_dim}")
    print(f"  gpt2_dim: {config.gpt2_dim}")
    print("✅ Config created successfully")
    return config

def test_bdh_with_sbert(config):
    """Test BDH model with SBERT embeddings."""
    print("\nTesting BDH with SBERT embeddings...")
    from bdh import BDH_GPU
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BDH_GPU(config).to(device)
    
    # Create fake SBERT embeddings [batch=2, seq=10, sbert_dim=384]
    batch_size = 2
    seq_len = 10
    fake_sbert_embeds = torch.randn(batch_size, seq_len, config.sbert_dim, device=device)
    
    # Forward pass
    logits, state = model.forward(inputs_embeds=fake_sbert_embeds)
    
    print(f"  Input shape: {fake_sbert_embeds.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Expected: [{batch_size}, {seq_len}, {config.vocab_size}]")
    
    # Get state representation
    state_rep = model.get_state_representation(state)
    print(f"  State representation shape: {state_rep.shape}")
    print(f"  Expected: [{batch_size}, {config.n_neurons}]")
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert state_rep.shape == (batch_size, config.n_neurons)
    
    print("✅ BDH forward pass with SBERT embeddings successful")
    return model

def test_rationale_decoder(config, bdh_model):
    """Test rationale decoder."""
    print("\nTesting RationaleDecoder...")
    from bdh import RationaleDecoder
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    decoder = RationaleDecoder(config, freeze_gpt2=True).to(device)
    
    # Create fake BDH state
    batch_size = 2
    fake_state = torch.randn(batch_size, config.n_neurons, device=device)
    
    # Get prefix embeddings
    prefix_embeds = decoder.get_prefix_embeddings(fake_state)
    print(f"  BDH state shape: {fake_state.shape}")
    print(f"  Prefix embeddings shape: {prefix_embeds.shape}")
    print(f"  Expected: [{batch_size}, {config.rationale_prefix_len}, {config.gpt2_dim}]")
    
    assert prefix_embeds.shape == (batch_size, config.rationale_prefix_len, config.gpt2_dim)
    
    print("✅ RationaleDecoder prefix generation successful")
    
    # Test generation (short)
    print("\n  Testing generation (this may take a moment)...")
    rationales = decoder.generate(fake_state, max_new_tokens=10)
    print(f"  Generated {len(rationales)} rationales")
    for i, r in enumerate(rationales):
        print(f"    Sample {i}: '{r[:50]}...'")
    
    print("✅ RationaleDecoder generation successful")
    return decoder

def test_sbert_encoder():
    """Test SBERT encoder (requires sentence-transformers)."""
    print("\nTesting SBERTEncoder...")
    try:
        from bdh import SBERTEncoder
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = SBERTEncoder(model_name='all-MiniLM-L6-v2', device=device)
        
        test_text = "This is a test sentence. Here is another one."
        embeddings = encoder.encode_text(test_text)
        
        print(f"  Input text: '{test_text}'")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Expected dim: {encoder.embedding_dim}")
        
        assert embeddings.shape[1] == encoder.embedding_dim
        
        print("✅ SBERTEncoder successful")
        return encoder
    except ImportError as e:
        print(f"⚠️ SBERTEncoder test skipped (missing dependency): {e}")
        return None

def test_end_to_end():
    """Full end-to-end test."""
    print("\n" + "="*60)
    print("END-TO-END TEST")
    print("="*60)
    
    try:
        from bdh import BDHConfig, BDH_GPU, SBERTEncoder, RationaleDecoder
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        
        # Create config
        config = BDHConfig(
            n_neurons=256,
            n_layers=2,
            n_heads=4,
            use_sbert=True
        )
        
        # Create models
        bdh = BDH_GPU(config).to(device)
        encoder = SBERTEncoder(device=device)
        decoder = RationaleDecoder(config, freeze_gpt2=True).to(device)
        
        # Test text 1: Contradiction
        backstory = "John is a brave knight who fights dragons."
        statement_contradict = "John was afraid of the small lizard."
        
        print(f"\n--- Case 1: Contradiction ---")
        print(f"Backstory: '{backstory}'")
        print(f"Statement: '{statement_contradict}'")
        
        # Encode
        backstory_emb = encoder.encode_text(backstory)
        stmt_bad_emb = encoder.encode_text(statement_contradict)
        full_emb_bad = torch.cat([backstory_emb, stmt_bad_emb], dim=0).unsqueeze(0).to(device)
        
        # Forward through BDH
        _, state_bad = bdh.forward(inputs_embeds=full_emb_bad)
        bdh_state_bad = bdh.get_state_representation(state_bad)
        
        # Generate rationale
        rationales_bad = decoder.generate(bdh_state_bad, max_new_tokens=20, prompt="This statement is ")
        print(f"Generated rationale: '{rationales_bad[0]}'")
        
        # Test text 2: Consistent
        statement_consistent = "John drew his sword and charged at the beast."
        
        print(f"\n--- Case 2: Consistent ---")
        print(f"Statement: '{statement_consistent}'")
        
        # Encode
        stmt_good_emb = encoder.encode_text(statement_consistent)
        full_emb_good = torch.cat([backstory_emb, stmt_good_emb], dim=0).unsqueeze(0).to(device)
        
        # Forward through BDH
        _, state_good = bdh.forward(inputs_embeds=full_emb_good)
        bdh_state_good = bdh.get_state_representation(state_good)
        
        # Generate rationale
        rationales_good = decoder.generate(bdh_state_good, max_new_tokens=20, prompt="This statement is ")
        print(f"Generated rationale: '{rationales_good[0]}'")
        
        print("\n" + "="*60)
        print("✅ END-TO-END TEST PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ END-TO-END TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("SBERT & GPT-2 Decoder Integration Tests")
    print("="*60)
    
    # Run tests
    try:
        test_imports()
        config = test_config()
        bdh_model = test_bdh_with_sbert(config)
        test_rationale_decoder(config, bdh_model)
        encoder = test_sbert_encoder()
        
        if encoder is not None:
            test_end_to_end()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
