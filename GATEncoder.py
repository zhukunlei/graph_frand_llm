from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch

class GATEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.gate1 = GATEncoder(input_dim, hidden_dim, num_heads)
        self.gate2 = GATEncoder(hidden_dim * num_heads, output_dim, num_heads=1)

    def forward(self, x, edge_index):
        h = F.elu(self.gate1(x, edge_index))
        h = F.elu(self.gate2(h, edge_index))
        return h.mean(dim=1)

    