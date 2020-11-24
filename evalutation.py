from torch_geometric.data import Batch, ClusterData
from torch_geometric.nn import DataParallel
import torch
import torch.nn.functional as F
import time
from torch.cuda import device_count
from Models import SimpleGCN
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader, DataListLoader


import argparse
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
    if model_name == 'simple_gcn': return SimpleGCN(dataset.num_features, dataset.num_classes)
    else:
        print('Model initialization failed')
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_gpu', type=str2bool, default=True, help='Toggle multi gpu, 1=True, 0=False')
    parser.add_argument('--use_ogb_dataset', type=str2bool, default=True, help='Use ogb dataset? 1=True, 0= False')
    parser.add_argument('--dataset_name', type=str, default='ogbn-products', help='Dataset name')
    parser.add_argument('--model', type=str, default='simple_gcn', help='Available models: simple_gcn, sage_gcn]')
    parser.add_argument('--subgraph_scheme', type=str, default='cluster', help='scheme of generating subgraphs')
    parser.add_argument('--num_parts', type=int, default=1500, help='number of clusters for cluster_gcn')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')

    args = parser.parse_args()
    dataset = PygNodePropPredDataset(name=args.dataset_name)
    data = dataset[0]
    print(f'data: {data}')

    if args.multi_gpu:
        print(f'Using {device_count()} GPUs!')

        # Prepare model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = parse_model_name(args.model, dataset)
        model = DataParallel(model)
        model = model.to(device)


        # Split data into subgraphs using cluster methods
        data_list = list(ClusterData(data, num_parts=args.num_parts))
        loader = DataListLoader(data_list, batch_size=100, shuffle=True)


        # Start training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



        for i in range(args.epochs):
            total_loss = 0
            for input_list in loader:
                output = model(input_list)
                y = torch.cat([data.y for data in input_list]).to(output.device).squeeze()
                print(y.size())
                loss = F.nll_loss(output, y.long())
                total_loss += loss
                loss.backward()
                optimizer.step()

            print(f'Epoch: {i} | Total: {total_loss}')










