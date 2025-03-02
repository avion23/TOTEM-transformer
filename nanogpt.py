import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=DROPOUT):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        
        self.proj = nn.Linear(model_dim, model_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        self.register_buffer("mask", torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)))
        
    def forward(self, x):
        B, T, C = x.size()
        
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.proj(y)
        y = self.proj_dropout(y)
        
        return y


class FeedForward(nn.Module):
    def __init__(self, model_dim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=DROPOUT):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = CausalSelfAttention(model_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ffwd = FeedForward(model_dim, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, model_dim=MODEL_DIM, num_heads=NUM_HEADS, 
                 num_layers=NUM_LAYERS, dropout=DROPOUT, context_length=CONTEXT_LENGTH):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, context_length, model_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, dropout) for _ in range(num_layers)])
        
        self.ln_f = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        idx = idx.contiguous()
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding[:, :T, :]
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            targets = targets.contiguous()
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        idx = idx.contiguous()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= CONTEXT_LENGTH else idx[:, -CONTEXT_LENGTH:].contiguous()
            
            logits, _ = self(idx_cond)
            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1).contiguous()
        
        return idx