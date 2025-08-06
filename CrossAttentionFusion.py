import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, graph_dim ,text_dim):
        self.graph_projection = nn.Linear(graph_dim,text_dim)
        self.query_projection = nn.Linear(text_dim,text_dim)
        self.key_projection = nn.Linear(text_dim,text_dim)
        self.value_projection = nn.Linear(text_dim,text_dim)

        self.cross_attention = nn.MultiheadAttention(text_dim,num_heads=8)

    def forward(self,graph_embedding,text_embedding):
        g = self.graph_projection(graph_embedding).unsqueeze(1)
        q = self.query_projection(g)
        k = self.key_projection(text_embedding)
        v = self.value_projection(text_embedding)

        fusion_res, _ = self.cross_attention(q,k,v)
        return fusion_res





