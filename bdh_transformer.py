import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BDHAttention(nn.Module):
    """
    GPT-2 Style BDH Attention with Temporal Gating.
    
    Implements the report's g function:
        S_t = (1 - E_t) ⊙ (α * S_{t-1}) + W_t ⊙ (K_t^T V_t)
    
    Where:
        - E_t: Erase gate (what to forget)
        - W_t: Write gate (how strongly to write new info)
        - α: Learnable decay rate
    
    Research Reference: 'Linear Transformers' (Katharopoulos et al.) 
    modified with BDH Bilinear Ingestion gates.
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.n_layer = config.n_layer
        
        # Projection for Q, K, V
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # State-Space Kernel: [Heads, dk, dv]
        self.register_buffer("state", torch.zeros(self.n_head, self.head_dim, self.head_dim))
        
        # === Temporal Gating (from report's g function) ===
        # Learnable decay rate α (per-head)
        self.alpha = nn.Parameter(torch.sigmoid(torch.randn(self.n_head, 1, 1)))
        
        # Erase gate: decides what to forget
        # Input: K features [n_head, head_dim], Output: gate [n_head, head_dim]
        self.erase_gate = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.Sigmoid()
        )
        
        # Write gate: decides how strongly to write new facts
        self.write_gate = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.Sigmoid()
        )
        
    def reset_state(self):
        self.state.zero_()

    def forward(self, x, use_state=False):
        B, T, C = x.size()
        head_dim = self.head_dim
        
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_head, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, nh, T, dk]
        
        # 1. Feature Mapping (Standard for Linear Transformers)
        # We use ELU+1 to ensure positivity as per original paper
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        if use_state and self.training:
            # Parallel Scan for Speed (Causal) with gating
            kv = k.unsqueeze(-1) * v.unsqueeze(-2) # [B, nh, T, dk, dv]
            s_seq = torch.cumsum(kv, dim=2) # [B, nh, T, dk, dv]
            
            # Apply Initial State if any
            s_seq = s_seq + self.state.view(1, self.n_head, 1, head_dim, head_dim)
            
            y = torch.matmul(q.unsqueeze(-2), s_seq).squeeze(-2) # [B, nh, T, dv]
            
            # Update the persistent buffer for ingestion logic (detach to keep graph clean)
            with torch.no_grad():
                self.state.copy_(s_seq[0, :, -1].detach())
                
        elif use_state:
            # Recurrent Mode (Inference/Ingestion) WITH GATING
            out = []
            curr_state = self.state.clone()
            
            for t in range(T):
                q_t = q[:, :, t, :].unsqueeze(-2) # [B, nh, 1, dk]
                k_t = k[:, :, t, :]  # [B, nh, dk]
                v_t = v[:, :, t, :].unsqueeze(-2) # [B, nh, 1, dv]
                
                # Compute gates from K (key represents incoming info)
                # k_t: [B, nh, dk] -> mean over batch -> [nh, dk]
                k_mean = k_t.mean(dim=0)  # [nh, dk]
                
                # Erase gate: how much to forget [nh, dk, 1]
                erase = self.erase_gate(k_mean).unsqueeze(-1)  # [nh, dk, 1]
                
                # Write gate: how strongly to write [nh, dk, 1]
                write = self.write_gate(k_mean).unsqueeze(-1)  # [nh, dk, 1]
                
                # KV outer product
                k_t_exp = k_t.unsqueeze(-1)  # [B, nh, dk, 1]
                kv = torch.matmul(k_t_exp, v_t)  # [B, nh, dk, dv]
                
                # GATED UPDATE: S = (1-E) * (α * S) + W * KV
                # This is the g function from the report!
                curr_state = (1 - erase) * (self.alpha * curr_state) + write * kv.mean(dim=0)
                
                # y = q * S
                y_t = torch.matmul(q_t, curr_state.unsqueeze(0))
                out.append(y_t.squeeze(-2))
            
            y = torch.stack(out, dim=2)
            self.state = curr_state.detach()
            
        else:
            # Standard Attention behavior if no state is used
            kv = k.unsqueeze(-1) * v.unsqueeze(-2)
            s_seq = torch.cumsum(kv, dim=2)
            y = torch.matmul(q.unsqueeze(-2), s_seq).squeeze(-2)
            
        # Reshape and project
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class BDHBlock(nn.Module):
    """
    Canonical GPT-2 Block with Pre-Norm and BDH-Attention.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = BDHAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x, use_state=False):
        x = x + self.attn(self.ln_1(x), use_state=use_state)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2BDHTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([BDHBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight sharing as per GPT-2
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_state(self):
        for block in self.transformer.h:
            block.attn.reset_state()

    def get_state(self):
        """Return list of [nh, dk, dv] tensors for each layer."""
        return [block.attn.state.clone() for block in self.transformer.h]
        
    def set_state(self, state_list):
        """Load list of states into layers."""
        for i, block in enumerate(self.transformer.h):
            block.attn.state.copy_(state_list[i])

    def forward(self, idx, targets=None, use_state=False):
        device = idx.device
        b, t = idx.size()
        
        # 1. Embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # 2. Sequential Blocks
        for block in self.transformer.h:
            x = block(x, use_state=use_state)
            
        # 3. Final Head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
