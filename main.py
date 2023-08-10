import argparse
import torch
import sys
import os
import models
import models_ogb
import trainers
import torch_geometric as pyg
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric import transforms
import numpy as np
import time


def ogb_prep(db, db_dir):
    dataset = PygNodePropPredDataset(name=db, root=db_dir)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    # the original graph is a directional graph, we should convert it into a bi-directional graph
    graph = dataset[0]
    label = graph.y

    # add reverse edges
    edge_index = graph.edge_index
    print(f"Total edges before adding reverse edges: {len(edge_index[0])}")
    reverse_edge_index = torch.cat([edge_index[1].unsqueeze(0), edge_index[0].unsqueeze(0)], dim=0)
    edge_index = torch.cat((edge_index, reverse_edge_index), dim=1)
    # remove duplicate edges
    edge_index = pyg.utils.coalesce(edge_index)
    graph.edge_index = edge_index
    print(f"Total edges after adding reverse edges: {len(edge_index[0])}")

    train_mask = torch.zeros(label.shape[0]).scatter_(0, train_idx, torch.ones(train_idx.shape[0])).bool()
    valid_mask = torch.zeros(label.shape[0]).scatter_(0, valid_idx, torch.ones(valid_idx.shape[0])).bool()
    test_mask = torch.zeros(label.shape[0]).scatter_(0, test_idx, torch.ones(test_idx.shape[0])).bool()
    graph.train_mask = train_mask
    graph.val_mask = valid_mask
    graph.test_mask = test_mask
    graph.y = label.squeeze()

    dataset[0] = graph
    return dataset

def load_data(db, db_dir='./dataset'):
    try:
        if db == 'ogbn-arxiv':
            data = ogb_prep('ogbn-arxiv', db_dir)
        else:
            data = Planetoid(db_dir, db, transform=transforms.NormalizeFeatures())
    except:
        raise ValueError('Unknown dataset: {}'.format(db))

    g = data[0]
    info_dict = {'in_dim': g.x.shape[1],
                 'out_dim': data.num_classes, }
    return g, info_dict

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return

def main(args):

    acc_list = []
    time_cost_list = []
    for i in range(args.round):
        set_seed(i)
        g, info_dict = load_data(args.dataset)
        info_dict.update(args.__dict__)
        info_dict.update({'device': torch.device('cpu') if args.gpu == -1 else torch.device('cuda:{}'.format(args.gpu)),})

        info_dict.update({'seed': i})

        # initialize a model
        model = getattr(models, args.model)(info_dict) if args.dataset != 'ogbn-arxiv' else getattr(models_ogb, args.model)(info_dict)
        # initialize a trainer
        backbone_list = ['GCN', 'GAT', 'SAGE', 'JKNet', 'GCN2', 'APPNPNet', 'GIN', 'SGC']
        if args.model in backbone_list:
            trainer = getattr(trainers, 'BaseTrainer')(g, model, info_dict)
        else:
            info_dict.update({'backbone': args.model[6:]})
            trainer = getattr(trainers, 'ViolinTrainer')(g, model, info_dict)

        model.to(info_dict['device'])
        print(model)
        print('\nSTART TRAINING\n')
        tic = time.time()
        val_acc, tt_acc, val_acc_fin, tt_acc_fin, microf1, macrof1 = trainer.train()
        toc = time.time()
        acc_list.append(tt_acc)
        time_cost = toc - tic
        time_cost_list.append(time_cost)
        print('The time cost of the {} round ({} epochs) is: {}.'.format(i, info_dict['n_epochs'], time_cost))

    print('\n\n')
    print('The averaged accuracy of {} rounds of experiments on {} is: {}'.format(args.round, args.dataset, np.mean(acc_list)))
    print('The averaged time cost (seconds/ 100 epochs) of {} rounds is {:.4f}'.format(args.round, np.mean(time_cost_list) / args.n_epochs * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='the main program to run experiments on small datasets')
    parser.add_argument("--round", type=int, default=10,
                        help="number of rounds to repeat the experiment")
    parser.add_argument("--model", type=str, default='ViolinGCN',
                        help="model name")
    parser.add_argument("--dataset", type=str, default='Cora',
                        help="the dataset for the experiment")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="the number of training epochs")
    parser.add_argument("--eta", type=int, default=1,
                        help="the interval (epoch) to override/ update the estimated labels")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="the number of hidden layers")
    parser.add_argument("--hid_dim", type=int, default=16,
                        help="the hidden dimension of hidden layers in the backbone model")
    parser.add_argument("--dropout", type=float, default=0.6,
                        help="dropout rate")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="specify the gpu index, set -1 to train on cpu")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="the learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="the weight decay for optimizer")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="coefficient for the consistency loss")
    parser.add_argument("--gamma", type=float, default=0.6,
                        help="coefficient for the virtual link loss")
    parser.add_argument("--cls_mode", type=str, default='virt',
                        help="the type of the classification loss (Eq.10), 'virt' only includes the second term, while 'both' inlcudes both terms")
    parser.add_argument("--delta", type=float, default=0.9,
                        help="the acc requirement (\delta) to pick node candidates for building VOs")
    parser.add_argument("--m", type=int, default=1,
                        help="the number of VOs per node")
    parser.add_argument("--bn", action='store_true', default=False,
                        help="a flag to indicate whether use batch-norm for training")
    args = parser.parse_args()

    main(args)