from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch import index_select
from torch_geometric.data import Data
class NeighborSubgraphLoader:
    def __init__(self, data, num_parts, shuffle=True):
        num_nodes = data.x.size()[0]
        batch_size = int(num_nodes/num_parts)
        self.neighbor_sampler_iter = iter(NeighborSampler(data.edge_index, sizes=[-1], batch_size=batch_size, shuffle=shuffle))
        self.data = data
    def __iter__(self):
        return self

    def __next__(self):
        batch_size, nid, edge_index = next(self.neighbor_sampler_iter)
        x = index_select(self.data.x, 0, nid)
        y = self.data.y[nid]
        edge_index_tensor = edge_index.edge_index
        ret = Data(x=x, edge_index=edge_index_tensor, y=y)
        return ret

