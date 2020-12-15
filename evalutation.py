from torch_geometric.data import Batch, ClusterData
from torch_geometric.nn import DataParallel
import torch
import torch.nn.functional as F
import time
from torch.cuda import device_count
from Models import SimpleGCN
from torch_geometric.datasets import Planetoid
import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader, DataListLoader
from NeighborSubgraphLoader import NeighborSubgraphLoader
import pickle as pk
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        return SimpleGCN(dataset.num_features, dataset.num_classes)
    else:
        print('Model initialization failed')
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_gpu', type=str2bool, default=True, help='Toggle multi gpu, 1=True, 0=False')
    parser.add_argument('--use_ogb_dataset', type=str2bool, default=True, help='Use ogb dataset? 1=True, 0= False')
    parser.add_argument('--dataset_name', type=str, default='ogbn-products', help='Dataset name')
    parser.add_argument('--model', type=str, default='simple_gcn', help='Available models: simple_gcn, sage_gcn]')
    parser.add_argument('--subgraph_scheme', type=str, default='neighbor', help='scheme of generating subgraphs')
    parser.add_argument('--num_parts', type=int, default=1000, help='number of clusters for cluster_gcn')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--neighbor_batch_size', type=int, default=256, help='Number of epochs')
    parser.add_argument('--data_list_batch_size', type=int, default=400, help='Number of epochs')

    args = parser.parse_args()
    dataset = PygNodePropPredDataset(name=args.dataset_name)
    data = dataset[0]
    # dataset = 'Cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    # dataset = Planetoid(path, dataset)
    # data = dataset[0]
    print(f'data: {data}')

    if args.multi_gpu:
        print(f'Using {device_count()} GPUs!')

        # Prepare model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        model = parse_model_name(args.model, dataset)
        model = DataParallel(model)
        model = model.to(device)

        if args.subgraph_scheme == 'cluster':
            # Split data into subgraphs using cluster methods
            data_list = list(ClusterData(data, num_parts=args.num_parts))

            loader = DataListLoader(data_list, batch_size=args.data_list_batch_size, shuffle=True)
        elif args.subgraph_scheme == 'neighbor':

            data_list = list(NeighborSubgraphLoader(data, batch_size=args.neighbor_batch_size))
            loader = DataListLoader(data_list, batch_size=args.data_list_batch_size, shuffle=True)
            print(f'Using neighbor sampling | number of subgraphs: {len(data_list)}')

        # Start training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

        running_times = []
        running_losses = []
        running_acc = []
        for i in range(args.epochs):
            total = 0
            total_loss = 0
            epoch_correct = 0
            t_start = time.time()
            for input_list in loader:
                batch_start = time.time()
                output = model(input_list)
                _, predicted = torch.max(output, 1)
                y = torch.cat([data.y for data in input_list]).to(output.device).squeeze()
                total += y.size()[0]

                epoch_correct += (predicted == y).sum().item()

                loss = F.nll_loss(output, y.long())
                total_loss += loss
                loss.backward()
                optimizer.step()
                batch_end = time.time()
                print(f'batch size: {len(input_list)} | batch time: {batch_end - batch_start}')

            lr_scheduler.step(total_loss)
            epoch_acc = epoch_correct / total
            running_acc.append(epoch_acc)
            t_end = time.time()

            running_times.append(t_end - t_start)
            running_losses.append(total_loss)
            print(f'Epoch time: {t_end - t_start}')

            print(f'Epoch: {i} | Total: {total_loss}')
        save_dict = {'running_losses': running_losses, 'running_times': running_times, 'running_acc': running_acc}
        with open(f'./gpu_{device_count()}_batch_size_{args.data_list_batch_size}_{args.subgraph_scheme}.pk',
                  'wb') as f:
            pk.dump(save_dict, f)











