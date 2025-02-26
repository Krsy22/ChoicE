import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from MoE import MoELayer
#from SparseMoE import *
class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads=64, n_groups=8):
        super(GroupQueryAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups

        assert d_model % n_heads == 0
        self.n_heads_groups = self.n_heads // self.n_groups
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, self.n_groups * self.head_dim)
        self.w_v = nn.Linear(d_model, self.n_groups * self.head_dim)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def expand(self, data):
        batch, time = data.shape[0], data.shape[2]
        data = data[:, :, None, :, :].expand(batch, self.n_groups, self.n_heads_groups, time,
                                             self.head_dim).contiguous()
        data = data.view(batch, self.n_groups * self.n_heads_groups, time, self.head_dim)
        return data  

    def forward(self, q, k, v, mask=None):
        
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        batch = q.shape[0]
        
        q = q.view(batch, -1, self.n_groups * self.n_heads_groups, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch, -1, self.n_groups, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, -1, self.n_groups, self.head_dim).permute(0, 2, 1, 3)

        
        k = self.expand(k)
        v = self.expand(v)

        
        score = q @ k.transpose(2, 3) / math.sqrt(self.head_dim)
        
        if mask is not None:
           
            mask = mask.unsqueeze(1).unsqueeze(2)  
            
            score = score.masked_fill(mask, float('-inf'))  

        
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, -1, self.d_model)

        
        output = self.w_combine(score)
        return output
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_experts, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.d_model = d_model  
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        #self.self_attn1 = GroupQueryAttention(d_model=256, n_heads=64, n_groups=8)
        self.MoE = MoELayer(num_experts, d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask):
        
        attn_output, _ = self.self_attn(src,src,src, key_padding_mask=src_key_padding_mask)

        src = src + self.dropout(attn_output)  
        src = self.norm1(src)  

        ff_output = self.MoE(src)
        
        src = src + self.dropout(ff_output)  
        src = self.norm2(src)  

        return src

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.d_model)  

    def forward(self, src, src_key_padding_mask):
        for layer in self.layers:
            src = layer(src, src_key_padding_mask)
        return self.norm(src)  



