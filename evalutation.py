from torch_geometric.data import Batch, ClusterData
from torch_geometric.nn import DataParallel
import torch
import torch.nn.functional as F
import time

def evaluate_model(model, data, epochs=1000, batch_method='cluster', num_clusters=100, multi_gpu=True):
    if multi_gpu:
        if batch_method == 'cluster':
            data_list = list(ClusterData(data, num_parts=num_clusters))

        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = DataParallel(model)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        optimizer.zero_grad()
        t0 = time.clock()
        print(f'start time: {t0}')
        for i in range(epochs):

            output = model(data_list)
            print(model.device_ids)
            y = torch.cat([data.y for data in data_list]).to(output.device)
            loss = F.nll_loss(output, y.long())
            print(f'loss: {loss}')
            loss.backward()
            optimizer.step()
        t1 = time.clock()
        print(f'end time: {t1}, time elapsed: {t1-t0}')
    else:

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        print(data)
        optimizer.zero_grad()
        t0 = time.clock()
        print(f'start time: {t0}')
        for i in range(epochs):
            output = model(data)
            y = data.y
            loss = F.nll_loss(output, y.long())
            print(f'loss: {loss}')
            loss.backward()
            optimizer.step()
        t1 = time.clock()
        print(f'end time: {t1}, time elapsed: {t1 - t0}')

