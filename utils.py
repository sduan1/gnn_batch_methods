from torch import nn
from Models import SimpleGCN, DeeperGCN
import argparse

#https://github.com/lightaime/deep_gcns_torch/blob/master/gcn_lib/sparse/torch_nn.py
def norm_layer(norm_type, nc):
    # normalization layer 1d
    norm = norm_type.lower()
    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc, elementwise_affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_model_name(model_name, dataset):
    if model_name == 'simple_gcn':
        print('Using simple gcn')
        return SimpleGCN(dataset.num_features, dataset.num_classes)
    if model_name == 'deeper_gcn':
        print('Using deeper gcn')
        return DeeperGCN(dataset.num_classes, dataset.num_features)
    else:
        print('Model initialization failed')
        return None