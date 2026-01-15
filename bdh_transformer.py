import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BDHAttention(nn.Module):
    """
    GPT-2 Style BDH Attention.
    Implements Causal Linear Attention with State-Space persistence.
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
        # In BDH, we use the property S_t = S_{t-1} + k_t^T v_t
        self.register_buffer("state", torch.zeros(self.n_head, self.head_dim, self.head_dim))
        
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
            # Parallel Scan for Speed (Causal)
            # Memory efficient O(T) training
            kv = k.unsqueeze(-1) * v.unsqueeze(-2) # [B, nh, T, dk, dv]
            s_seq = torch.cumsum(kv, dim=2) # [B, nh, T, dk, dv]
            
            # Apply Initial State if any
            s_seq = s_seq + self.state.view(1, self.n_head, 1, head_dim, head_dim)
            
            y = torch.matmul(q.unsqueeze(-2), s_seq).squeeze(-2) # [B, nh, T, dv]
            
            # Update the persistent buffer for ingestion logic (detach to keep graph clean)
            with torch.no_grad():
                self.state.copy_(s_seq[0, :, -1].detach())
                
        elif use_state:
            # Recurrent Mode (Inference/Ingestion)
            out = []
            curr_state = self.state
            for t in range(T):
                q_t = q[:, :, t, :].unsqueeze(-2) # [B, nh, 1, dk]
                k_t = k[:, :, t, :].unsqueeze(-1) # [B, nh, dk, 1]
                v_t = v[:, :, t, :].unsqueeze(-2) # [B, nh, 1, dv]
                
                # S = S + k^T v
                curr_state = curr_state + torch.matmul(k_t, v_t)
                
                # y = q * S
                y_t = torch.matmul(q_t, curr_state)
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
