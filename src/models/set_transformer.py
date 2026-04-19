import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_k, self.n_heads = d_model // n_heads, n_heads
        self.W_q, self.W_k, self.W_v, self.W_o = [nn.Linear(d_model, d_model) for _ in range(4)]
        self.dropout = nn.Dropout(dropout)
    def forward(self, Q, K, V, mask=None):
        B, n_q, _ = Q.shape; _, n_kv, _ = K.shape
        q = self.W_q(Q).view(B, n_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, n_kv, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, n_kv, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, n_q, -1)
        return self.W_o(out), attn_weights

class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
    def forward(self, X, mask=None):
        attn_out, _ = self.mha(self.norm1(X), self.norm1(X), self.norm1(X), mask)
        X = X + attn_out
        X = X + self.ffn(self.norm2(X))
        return X, None

class PoolingByMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_seeds, dropout=0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.01)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout); self.norm = nn.LayerNorm(d_model)
    def forward(self, Z, mask=None):
        seeds_expanded = self.seeds.expand(Z.shape[0], -1, -1)
        out, _ = self.mha(self.norm(seeds_expanded), self.norm(Z), self.norm(Z), mask)
        return seeds_expanded + out, None

class SetTransformerDOT(nn.Module):
    def __init__(self, feat_dim, n_components, d_model=32, n_heads=4, n_layers=2, d_ff=64, n_seeds=3, dropout=0.25, comp_embed_dim=8):
        super().__init__()
        self.comp_embedding = nn.Embedding(n_components + 1, comp_embed_dim, padding_idx=0)
        self.input_proj = nn.Sequential(nn.Linear(feat_dim + comp_embed_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.sab_layers = nn.ModuleList([SetAttentionBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.pma = PoolingByMultiHeadAttention(d_model, n_heads, n_seeds, dropout)
        head_in_dim = (d_model * n_seeds) + 4
        self.shared_mlp = nn.Sequential(nn.Linear(head_in_dim, head_in_dim), nn.LayerNorm(head_in_dim), nn.GELU(), nn.Dropout(dropout))
        self.head_visc = nn.Sequential(nn.Linear(head_in_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.head_oxid = nn.Sequential(nn.Linear(head_in_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
    def forward(self, x, comp_ids, global_feats, mask=None):
        h = self.input_proj(torch.cat([x, self.comp_embedding(comp_ids)], dim=-1))
        if mask is not None: h = h * mask.unsqueeze(-1).float()
        for sab in self.sab_layers:
            h, _ = sab(h, mask)
            if mask is not None: h = h * mask.unsqueeze(-1).float()
        pooled, _ = self.pma(h, mask); pooled_flat = pooled.view(pooled.shape[0], -1)
        shared_features = self.shared_mlp(torch.cat([pooled_flat, global_feats], dim=-1))
        return torch.cat([self.head_visc(shared_features), self.head_oxid(shared_features)], dim=-1), None