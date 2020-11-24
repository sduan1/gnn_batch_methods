import os.path as osp
import torch
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.datasets import PPI, Planetoid, Reddit
import torch.nn.functional as F
class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, output_dim)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)