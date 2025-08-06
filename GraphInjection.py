import torch
import torch.nn as nn
from transformers import Qwen3MoeForCausalLM, Qwen3MoeConfig

class GraphInjection(nn.Module):
    def __init__(self, graph_dim, hidden_dim):
        super(GraphInjection, self).__init__()
        self.graph_projection = nn.Linear(graph_dim, hidden_dim)

    def forward(self, graph_embedding, hidden_state):
        """
        :param graph_embedding: (batch_size,graph_dim)
        :param hidden_state: (batch_size,seq_length,hidden_dim)
        :return:
        """
        graph_embedding = self.graph_projection(graph_embedding)
        graph_hidden = graph_embedding.unsqueeze(1).expand_as(hidden_state)
        return hidden_state + graph_hidden


