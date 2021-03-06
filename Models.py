import os.path as osp
import torch
from torch_geometric.nn import GCNConv, SAGEConv
from utils import norm_layer
import time
from torch_geometric.nn import GENConv, DeepGCNLayer
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging

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

#https://arxiv.org/abs/1904.03751
class DeeperGCN(torch.nn.Module):
    def __init__(self,
                 num_tasks,
                 in_channels,
                 hidden_channels=128,
                 num_layers=40,
                 dropout=0.5,
                 block='res+',
                 conv='gen',
                 aggr='softmax',
                 t=0.1,
                 p=None,
                 learn_t=None,
                 learn_p=None,
                 msg_norm=None,
                 learn_msg_scale=None,
                 norm='batch',
                 mlp_layers=1
                 ):
        super(DeeperGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.block = block

        self.checkpoint_grad = False

        t = t
        self.learn_t = learn_t
        p = p
        self.learn_p = learn_p
        self.msg_norm = msg_norm
        learn_msg_scale = learn_msg_scale

        norm = norm
        mlp_layers = mlp_layers

        if aggr in ['softmax_sg', 'softmax', 'power'] and self.num_layers > 3:
            self.checkpoint_grad = True
            self.ckp_k = self.num_layers // 2

        print('The number of layers {}'.format(self.num_layers),
              'Aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))

        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.node_features_encoder = torch.nn.Linear(in_channels, hidden_channels)
        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

        for layer in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              norm=norm)
            else:
                raise Exception('Unknown Conv Type')

            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

    def forward(self, data):
        start_time = time.time()
        x, edge_index = data.x, data.edge_index
        h = self.node_features_encoder(x)

        if self.block == 'res+':

            h = self.gcns[0](h, edge_index)

            if self.checkpoint_grad:

                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)

                    if layer % self.ckp_k != 0:
                        res = checkpoint(self.gcns[layer], h2, edge_index)
                        h = res + h
                    else:
                        h = self.gcns[layer](h2, edge_index) + h

            else:
                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    h = self.gcns[layer](h2, edge_index) + h

            h = F.relu(self.norms[self.num_layers - 1](h))
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = F.relu(h2)
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')

        h = self.node_pred_linear(h)

        end_time = time.time()

        print(f'Model processed {data.num_graphs} subgraphs | Model process time: {end_time-start_time}')
        return torch.log_softmax(h, dim=-1)

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))