import os.path as osp
import torch
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.datasets import PPI, Planetoid, Reddit
import torch.nn.functional as F
import torch_geometric.transforms as T


dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Reddit(path)
rawdata = dataset[0]
num_features = dataset.num_features
num_classes = dataset.num_classes
print(rawdata)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(num_features, 16)
        self.conv2 = SAGEConv(16, num_classes)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

from evalutation import evaluate_model
model = Net()
evaluate_model(model, rawdata)