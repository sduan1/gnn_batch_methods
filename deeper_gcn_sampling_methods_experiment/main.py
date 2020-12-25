import __init__
from ogb.nodeproppred import Evaluator
import torch
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from utils.data_util import intersection, random_partition_graph, generate_sub_graphs
from args import ArgsInit
from ogb.nodeproppred import PygNodePropPredDataset
from model import DeeperGCN
import numpy as np
from utils.ckpt_util import save_ckpt
import logging
import statistics
import time

from sklearn.metrics import accuracy_score
from torch_geometric.data import Data, RandomNodeSampler, GraphSAINTRandomWalkSampler, ClusterData
import pickle as pk
from NeighborSubgraphLoader import NeighborSubgraphLoader

@torch.no_grad()
def test(model, data_list):
    # test on CPU
    model.eval()
    model.to('cpu')

    y_pred_list = []
    y_true_list = []
    for sub_graph in data_list:
        out = model(sub_graph.x, sub_graph.edge_index)
        y_pred = out.argmax(dim=-1, keepdim=True).squeeze()
        target = sub_graph.y.squeeze()

        y_pred_list.append(y_pred)
        y_true_list.append(target)

    y_pred_list = torch.cat(y_pred_list)
    y_true_list = torch.cat(y_true_list)
    test_acc = accuracy_score(y_true_list, y_pred_list)

    # train_acc = evaluator.eval({
    #     'y_true': y_true[split_idx['train']],
    #     'y_pred': y_pred[split_idx['train']],
    # })['acc']
    # valid_acc = evaluator.eval({
    #     'y_true': y_true[split_idx['valid']],
    #     'y_pred': y_pred[split_idx['valid']],
    # })['acc']
    # test_acc = evaluator.eval({
    #     'y_true': y_true[split_idx['test']],
    #     'y_pred': y_pred[split_idx['test']],
    # })['acc']

    return test_acc


def train(data_list, model, optimizer, device):
    loss_list = []
    acc_list = []
    model.train()

    # sg_nodes, sg_edges = data
    # train_y = y_true[train_idx].squeeze(1)

    # idx_clusters = np.arange(len(sg_nodes))
    # np.random.shuffle(idx_clusters)

    y_pred_list = []
    y_true_list = []
    for sub_graph in data_list:
        print(sub_graph)

        # x_ = x[sg_nodes[idx]].to(device)
        # sg_edges_ = sg_edges[idx].to(device)
        # mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
        #
        # inter_idx = intersection(sg_nodes[idx], train_idx)
        # training_idx = [mapper[t_idx] for t_idx in inter_idx]

        optimizer.zero_grad()

        pred = model(sub_graph.x, sub_graph.edge_index)
        target = sub_graph.y.squeeze().to(device)
        y_pred = pred.argmax(dim=-1, keepdim=True).squeeze()

        y_true_list.append(target)
        y_pred_list.append(y_pred)

        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        batch_acc = accuracy_score(target, y_pred)
        print(f'batch_acc: {batch_acc}')
        acc_list.append(batch_acc)

    y_pred_list = torch.cat(y_pred_list)
    y_true_list = torch.cat(y_true_list)
    epoch_acc = accuracy_score(y_true_list, y_pred_list)

    temp_res = {}
    temp_res['acc'] = acc_list
    temp_res['loss'] = loss_list
    with open('neighbor_10.pk', 'wb') as f:
        pk.dump(temp_res, f)

    return statistics.mean(loss_list), epoch_acc


def main():

    EPOCHS = 1
    NUMBER_OF_SUBGRAPHS = 10
    CKPT_PATH = f'neighbor_deeper_num_{NUMBER_OF_SUBGRAPHS}.pt'
    EXPERIMENT_RES_PATH = f'neighbor_deeper_num_{NUMBER_OF_SUBGRAPHS}_experiment_res.pk'
    args = ArgsInit().args

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    dataset = PygNodePropPredDataset(name=args.dataset)
    graph = dataset[0]
    print(graph)
    num_parts = NUMBER_OF_SUBGRAPHS
    data_list = list(NeighborSubgraphLoader(graph, num_parts=NUMBER_OF_SUBGRAPHS))
    print(f'len of datalist: {len(data_list)}')
    number_of_train = int(0.9*num_parts)

    train_data_list = data_list[0:number_of_train]
    test_data_list = data_list[number_of_train:]

    print(f'Train test split successful, number of train: {len(train_data_list)} | number of test: {len(test_data_list)}')

    # adj = SparseTensor(row=graph.edge_index[0],
    #                    col=graph.edge_index[1])

    # if args.self_loop:
    #     adj = adj.set_diag()
    #     graph.edge_index = add_self_loops(edge_index=graph.edge_index,
    #                                       num_nodes=graph.num_nodes)[0]
    # split_idx = dataset.get_idx_split()
    # train_idx = split_idx["train"].tolist()


    #sub_dir = 'random-train_{}-full_batch_test'.format(args.cluster_number)
    #logging.info(sub_dir)

    args.in_channels = graph.x.size(-1)
    args.num_tasks = dataset.num_classes

    logging.info('%s' % args)

    model = DeeperGCN(args).to(device)

    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()
    epoch_loss_list = []
    epoch_acc_list = []
    highest_acc = 0
    best_model_dict = None
    for epoch in range(1, EPOCHS + 1):
        # generate batches
        # parts = random_partition_graph(graph.num_nodes,
        #                                cluster_number=args.cluster_number)
        # data = generate_sub_graphs(adj, parts, cluster_number=args.cluster_number)


        epoch_loss, epoch_acc= train(data_list, model, optimizer, device)
        epoch_loss_list.append(epoch_loss)
        epoch_acc_list.append(epoch_acc)
        print('Epoch {}, training loss {:.4f} | training acc {}'.format(epoch, epoch_loss, epoch_acc))

        test_acc = test(model, test_data_list)
        if test_acc > highest_acc:
            highest_acc = test_acc
            best_model_dict = model.state_dict()


        logging.info(f'best test acc: {highest_acc} | saved to path {CKPT_PATH}')



    #
    # logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))

    experiment_result = {}
    experiment_result['Total training time'] = total_time
    experiment_result['Epoch loss list'] = epoch_loss_list
    experiment_result['Epoch acc list'] = epoch_acc_list
    experiment_result['Best test acc'] = highest_acc

    torch.save(best_model_dict, CKPT_PATH)
    with open(EXPERIMENT_RES_PATH, 'wb') as f:
        pk.dump(experiment_result, f)



def test_model(model_path):
    args = ArgsInit().args
    dataset = PygNodePropPredDataset(name=args.dataset)
    graph = dataset[0]

    num_parts = 10
    data_list = list(RandomNodeSampler(graph, num_parts=num_parts, shuffle=True))
    number_of_train = int(0.9 * num_parts)

    train_data_list = data_list[0:number_of_train]
    test_data_list = data_list[number_of_train:]

    args.in_channels = graph.x.size(-1)
    args.num_tasks = dataset.num_classes

    model = DeeperGCN(args)
    model.load_state_dict(torch.load(model_path))

    print(test(model, test_data_list))

if __name__ == "__main__":
    main()


    #Test model
    # test_model('./saint_rw_deeper_num_1000.pt')