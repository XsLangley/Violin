import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.enc = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=info_dict['dropout'])
        for i in range(info_dict['n_layers']):
            in_dim = info_dict['in_dim'] if i == 0 else info_dict['hid_dim']
            out_dim = info_dict['out_dim'] if i == info_dict['n_layers'] - 1 else info_dict['hid_dim']
            bn = False if i == (info_dict['n_layers'] - 1) else info_dict['bn']
            self.enc.append(GCNConv(in_dim, out_dim))
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())

    def forward(self, x, edge_index):
        for i in range(self.info_dict['n_layers']):
            x = self.dropout(x)
            h = self.enc[i](x, edge_index)
            if i < self.info_dict['n_layers'] - 1:
                h = self.bns[i](h)
                h = self.act(h)
            x = h
        return x

    def reset_parameters(self):
        for i in range(self.info_dict['n_layers']):
            self.enc[i].reset_parameters()
            self.bns[i].reset_parameters()
