from torch_geometric.data import Batch, ClusterData
from torch_geometric.nn import DataParallel
import torch.nn.functional as F
import time
from Models import SimpleGCN, DeeperGCN
from ogb.nodeproppred import PygNodePropPredDataset
from NeighborSubgraphLoader import NeighborSubgraphLoader
import pickle as pk
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from unit_tests import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_gpu', type=str2bool, default=True, help='Toggle multi gpu, 1=True, 0=False')
    parser.add_argument('--use_ogb_dataset', type=str2bool, default=True, help='Use ogb dataset? 1=True, 0= False')
    parser.add_argument('--dataset_name', type=str, default='ogbn-products', help='Dataset name')
    parser.add_argument('--model', type=str, default='deeper_gcn', help='Available models: simple_gcn, sage_gcn]')
    parser.add_argument('--subgraph_scheme', type=str, default='neighbor', help='scheme of generating subgraphs')
    parser.add_argument('--num_parts', type=int, default=1000, help='number of clusters for cluster_gcn')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--neighbor_batch_size', type=int, default=256, help='Number of epochs')
    parser.add_argument('--data_list_batch_size', type=int, default=400, help='Number of epochs')

    args = parser.parse_args()

    # Load dataset

    dataset = PygNodePropPredDataset(name=args.dataset_name)
    data = dataset[0]
    dataset_test(data)

    if args.multi_gpu:
        # Unit test: GPU number verification


        # Prepare model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = parse_model_name(args.model, dataset)
        model = DataParallel(model)
        model = model.to(device)

        #Split graph into subgraphs
        if args.subgraph_scheme == 'cluster':
            # Split data into subgraphs using cluster methods
            data_list = list(ClusterData(data, num_parts=args.num_parts))
        elif args.subgraph_scheme == 'neighbor':
            data_list = list(NeighborSubgraphLoader(data, batch_size=args.neighbor_batch_size))
            print(f'Using neighbor sampling | number of subgraphs: {len(data_list)}')


        # Run the model for each batch size setups
        batch_sizes = np.array(list(range(1, 65)))*4
        batch_running_time = []
        for batch_size in batch_sizes:
            batch_size = int(batch_size)
            loader = DataListLoader(data_list, batch_size=batch_size, shuffle=True)

            # Model hyperparameters
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)


            temp_batch_times = []
            for input_list in loader:
                batch_start = time.time()
                #forward
                output = model(input_list)

                model_output_test(output, input_list)

                _, predicted = torch.max(output, 1)
                y = torch.cat([data.y for data in input_list]).to(output.device).squeeze()
                loss = F.nll_loss(output, y.long())

                #backward
                loss.backward()
                optimizer.step()
                batch_end = time.time()
                batch_time = batch_end - batch_start
                temp_batch_times.append(batch_time)
                print(f'batch size: {len(input_list)} | batch time: {batch_time}')

            average_batch_time = sum(temp_batch_times)/len(temp_batch_times)
            batch_running_time.append(average_batch_time)


        #Save experiment results
        res = {'batch_sizes':batch_sizes, 'batch_time':batch_running_time}
        with open(f'{device_count()}GPU_running_time_{args.model}.pk', 'wb') as f:
            pk.dump(res, f)