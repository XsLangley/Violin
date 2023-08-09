import torch
from torch.nn import functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, info_dict, use_linear=False):
        super().__init__()
        self.info_dict = info_dict
        self.n_layers = info_dict['n_layers']
        self.use_linear = use_linear

        self.enc = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(info_dict['n_layers']):
            in_dim = info_dict['hid_dim'] if i > 0 else info_dict['in_dim']
            out_dim = info_dict['hid_dim'] if i < info_dict['n_layers'] - 1 else info_dict['out_dim']
            bias = i == info_dict['n_layers'] - 1
            bn = False if i == (info_dict['n_layers'] - 1) else info_dict['bn']

            self.enc.append(GCNConv(in_dim, out_dim, bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_dim, out_dim, bias=False))
            self.norms.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())

        self.input_drop = nn.Dropout(min(0.1, info_dict['dropout']))
        self.dropout = nn.Dropout(info_dict['dropout'])
        self.activation = F.relu

    def forward(self, x, edge_index, virt_edge_index=None):
        x = self.input_drop(x)

        for i in range(self.n_layers):
            h = self.enc[i](x, edge_index)
            if virt_edge_index is not None:
                # compute virtual link features
                vh = self.enc[i](x, virt_edge_index)
                if self.info_dict['virt_agg'] == 'max':
                    h = torch.maximum(h, vh)
                elif self.info_dict['virt_agg'] == 'sum':
                    h = vh + h
                else:
                    raise ValueError('Unknown virtual link features aggregator!')

            if self.use_linear:
                linear = self.linear[i](x)
                h = h + linear

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

            x = h

        return x

    def reset_parameters(self):
        for i in range(self.n_layers):
            self.enc[i].reset_parameters()
            self.norms[i].reset_parameters()
            if self.use_linear:
                self.linear[i].reset_parameters()

class ViolinGCN(GCN):
    def __init__(self, info_dict):
        super().__init__(info_dict)
