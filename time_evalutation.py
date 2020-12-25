from torch_geometric.data import Batch, ClusterData
from torch_geometric.nn import DataParallel
import torch
import torch.nn.functional as F
import time
from torch.cuda import device_count
from Models import SimpleGCN, DeeperGCN
from torch_geometric.datasets import Planetoid
import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader, DataListLoader
from NeighborSubgraphLoader import NeighborSubgraphLoader
import pickle as pk
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle as pk
from utils import *
from unit_tests import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_gpu', type=str2bool, default=True, help='Toggle multi gpu, 1=True, 0=False')
    parser.add_argument('--use_ogb_dataset', type=str2bool, default=True, help='Use ogb dataset? 1=True, 0= False')
    parser.add_argument('--dataset_name', type=str, default='ogbn-products', help='Dataset name')
    parser.add_argument('--model', type=str, default='deeper_gcn', help='Available models: simple_gcn, sage_gcn]')
    parser.add_argument('--subgraph_scheme', type=str, default='neighbor', help='scheme of generating subgraphs')
    parser.add_argument('--num_parts', type=int, default=1000, help='number of clusters for cluster_gcn')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--neighbor_batch_size', type=int, default=128, help='Number of epochs')
    parser.add_argument('--data_list_batch_size', type=int, default=4, help='Number of epochs')
    parser.add_argument('--debug_mode', type=str2bool, default=True, help='Toggle debug mode')
    parser.add_argument('--num_gpu', type=int, default=1, help='number of gpus')

    args = parser.parse_args()
    dataset = PygNodePropPredDataset(name=args.dataset_name)
    data = dataset[0]


    if args.multi_gpu:
        gpu_test(args.num_gpu)

        # Prepare model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        model = parse_model_name(args.model, dataset)
        model = DataParallel(model)
        model = model.to(device)

        if args.subgraph_scheme == 'cluster':
            # Split data into subgraphs using cluster methods
            data_list = list(ClusterData(data, num_parts=args.num_parts))
            print(f'using cluster method')

        elif args.subgraph_scheme == 'neighbor':
            if args.debug_mode:
                # Use a smaller dataset for debug purpose
                with open('debug_saved_list.pk', 'rb') as f:
                    data_list = pk.load(f)
            else:
                data_list = list(NeighborSubgraphLoader(data, batch_size=args.neighbor_batch_size))


            print(f'Using neighbor sampling | number of subgraphs: {len(data_list)}')

        loader = DataListLoader(data_list, batch_size=args.data_list_batch_size, shuffle=True)
        # Training

        # Hyperparameters
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
            for i, input_list in enumerate(loader):
                print(f"*********************Batch{i} info*********************")
                print(f'input_list len: {len(input_list)}')
                batch_start = time.time()
                forward_start = time.time()
                output = model(input_list)
                forward_end = time.time()
                print(f'forward passing time: {forward_end - forward_start}')
                _, predicted = torch.max(output, 1)
                y = torch.cat([data.y for data in input_list]).to(output.device).squeeze()
                total += y.size()[0]

                epoch_correct += (predicted == y).sum().item()

                loss = F.nll_loss(output, y.long())
                total_loss += loss

                backward_start = time.time()
                loss.backward()
                backward_end = time.time()
                print(f'backward_time: {backward_end - backward_start}')

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











