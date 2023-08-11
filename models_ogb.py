import torch
from torch.nn import functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GINConv
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul


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


class GAT(nn.Module):
    def __init__(self, info_dict, use_attn_dst=True):
        super().__init__()
        self.info_dict = info_dict
        self.n_layers = info_dict['n_layers']
        self.use_attn_dst = use_attn_dst

        self.enc = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(info_dict['n_layers']):
            in_dim = info_dict['num_heads'] * info_dict['hid_dim'] if i > 0 else info_dict['in_dim']
            out_dim = info_dict['hid_dim'] if i < info_dict['n_layers'] - 1 else info_dict['out_dim']
            num_heads = info_dict['num_heads'] if i < info_dict['n_layers'] - 1 else 1
            bn = False if i == (info_dict['n_layers'] - 1) else info_dict['bn']
            self.enc.append(GATConv(in_dim, out_dim, heads=num_heads, dropout=info_dict['attn_drop']))
            self.norms.append(nn.BatchNorm1d(num_heads * out_dim) if bn else nn.Identity())

            
        self.input_drop = nn.Dropout(info_dict['input_drop'])
        self.dropout = nn.Dropout(info_dict['dropout'])
        self.activation = F.relu

    def forward(self, x, edge_index):
        x = self.input_drop(x)

        for i in range(self.n_layers):
            if not self.use_attn_dst:
                x = (x, None)
            h = self.enc[i](x, edge_index)
            

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

            x = h

        return x

    def reset_param(self):
        for i in range(self.n_layers):
            self.enc[i].reset_parameters()
            self.norms[i].reset_parameters()


class ViolinGAT(GAT):
    def __init__(self, info_dict):
        super().__init__(info_dict)


class SAGE(nn.Module):
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

            self.enc.append(SAGEConv(in_dim, out_dim, bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_dim, out_dim, bias=False))
            self.norms.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())

        self.input_drop = nn.Dropout(min(0.1, info_dict['dropout']))
        self.dropout = nn.Dropout(info_dict['dropout'])
        self.activation = F.relu

    def forward(self, x, edge_index):
        x = self.input_drop(x)

        for i in range(self.n_layers):
            h = self.enc[i](x, edge_index)
            
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


class ViolinSAGE(SAGE):
    def __init__(self, info_dict):
        super().__init__(info_dict)


class JKNet(nn.Module):
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
            out_dim = info_dict['hid_dim']
            bias = i == info_dict['n_layers'] - 1
            bn = False if i == (info_dict['n_layers'] - 1) else info_dict['bn']
            self.enc.append(SAGEConv(in_dim, out_dim, bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_dim, out_dim, bias=False))
            self.norms.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())

        self.input_drop = nn.Dropout(min(0.1, info_dict['dropout']))
        self.dropout = nn.Dropout(info_dict['dropout'])
        self.activation = F.relu
        # the classifier for the final output head
        rep_dim = info_dict['hid_dim'] * info_dict['n_layers']
        self.classifier = nn.Linear(rep_dim, info_dict['out_dim'])

    def forward(self, x, edge_index):
        feat_list = []
        x = self.input_drop(x)

        for i in range(self.n_layers):
            h = self.enc[i](x, edge_index)
            
            if self.use_linear:
                linear = self.linear[i](x)
                h = h + linear

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

            feat_list.append(h)
            x = h

        x = torch.cat(feat_list, dim=-1)
        x = self.classifier(x)

        return x

    def reset_parameters(self):
        for i in range(self.n_layers):
            self.enc[i].reset_parameters()
            self.norms[i].reset_parameters()
            if self.use_linear:
                self.linear[i].reset_parameters()


class ViolinJKNet(JKNet):
    def __init__(self, info_dict):
        super().__init__(info_dict)


class SGC(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.enc = SGConv(info_dict['in_dim'], info_dict['out_dim'], K=info_dict['n_layers'], cached=True)

    def forward(self, x, edge_index):
        h = self.enc(x, edge_index)
        return h

    def reset_parameters(self):
        for i in range(self.info_dict['n_layers']):
            self.enc.reset_parameters()


class GCNIIdenseConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = True,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias='bn', **kwargs):

        super(GCNIIdenseConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(in_channels, out_channels))
        if bias == 'bn':
            self.norm = nn.BatchNorm1d(out_channels)
        elif bias == 'ln':
            self.norm = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index, alpha, h0,
                edge_weight=None):
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        support = x + torch.matmul(x, self.weight1)
        initial = alpha * h0 + torch.matmul(h0, self.weight2)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=support, edge_weight=edge_weight,
                             size=None) + initial

        out = self.norm(out)
        return out

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        assert edge_weight is not None
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCN2(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.n_layers = info_dict['n_layers']

        self.enc = nn.ModuleList()

        for i in range(info_dict['n_layers'] + 2):
            in_dim = info_dict['hid_dim'] if i > 0 else info_dict['in_dim']
            out_dim = info_dict['hid_dim'] if i < info_dict['n_layers'] + 1 else info_dict['out_dim']
            bias = 'bn' if info_dict['bn'] else None
            if (i == 0) or (i == info_dict['n_layers'] + 1):
                self.enc.append(nn.Linear(in_dim, out_dim))
            else:
                self.enc.append(GCNIIdenseConv(in_dim, out_dim, bias=bias))

        self.dropout = nn.Dropout(info_dict['dropout'])
        self.activation = F.relu
        self.alpha = 0.5

    def forward(self, x, edge_index):
        _hidden = []
        # input transformation
        x = self.dropout(x)
        x = self.activation(self.enc[0](x))
        _hidden.append(x)

        # graph convolution
        for i in range(1, self.n_layers + 1):
            x = self.dropout(x)
            h = self.enc[i](x, edge_index, self.alpha, _hidden[0])
            h = self.activation(h) + _hidden[-1]
            _hidden.append(h)
            x = h

        x = self.dropout(x)
        x = self.enc[-1](x)

        return x

    def reset_parameters(self):
        for i in range(self.n_layers):
            self.enc[i].reset_parameters()


class ViolinGCN2(GCN2):
    def __init__(self, info_dict):
        super().__init__(info_dict)


class GIN(nn.Module):
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
            bn = False if i == (info_dict['n_layers'] - 1) else info_dict['bn']

            self.enc.append(GINConv(nn.Linear(in_dim, out_dim), train_eps=True,))
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


class ViolinGIN(GIN):
    def __init__(self, info_dict):
        super().__init__(info_dict)
