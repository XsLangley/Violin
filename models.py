import torch
from torch import nn
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, SGConv, SAGEConv, APPNP, GINConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


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


class GAT(nn.Module):
    def __init__(self, info_dict):
        super(GAT, self).__init__()
        self.info_dict = info_dict
        self.enc = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = nn.ELU()
        self.dropout = nn.Dropout(p=info_dict['dropout'])

        for l in range(info_dict['n_layers']):
            in_dim = info_dict['in_dim'] if l == 0 else info_dict['hid_dim'] * info_dict['num_heads']
            out_dim = info_dict['out_dim'] if l == info_dict['n_layers'] - 1 else info_dict['hid_dim']
            num_heads = info_dict['num_heads'] if l < info_dict['n_layers'] - 1 else 1
            concat = True if l < info_dict['n_layers'] - 1 else False
            bn = False if l == (info_dict['n_layers'] - 1) else info_dict['bn']
            self.enc.append(GATConv(in_dim, out_dim, num_heads, concat=concat, dropout=info_dict['attn_drop']))
            self.bns.append(nn.BatchNorm1d(num_heads * out_dim) if bn else nn.Identity())

    def forward(self, x, edge_index):

        for l in range(self.info_dict['n_layers']):
            x = self.dropout(x)
            h = self.enc[l](x, edge_index)
            if l < self.info_dict['n_layers'] - 1:
                h = self.bns[l](h)
                h = self.act(h)

            x = h

        return x

    def reset_param(self):
        for i in range(self.info_dict['n_layers']):
            self.enc[i].reset_parameters()
            self.bns[i].reset_parameters()


class SAGE(nn.Module):
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
            self.enc.append(SAGEConv(in_dim, out_dim))
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


class JKNet(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.enc = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=info_dict['dropout'])
        for i in range(info_dict['n_layers']):
            in_dim = info_dict['in_dim'] if i == 0 else info_dict['hid_dim']
            out_dim = info_dict['hid_dim']
            bn = info_dict['bn']
            self.enc.append(GCNConv(in_dim, out_dim))
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())

        # the classifier for the final output head
        rep_dim = info_dict['hid_dim'] * info_dict['n_layers']
        self.classifier = nn.Linear(rep_dim, info_dict['out_dim'])

    def forward(self, x, edge_index):
        feat_list = []
        for i in range(self.info_dict['n_layers']):
            x = self.dropout(x)
            h = self.enc[i](x, edge_index)
            h = self.bns[i](h)
            h = self.act(h)
            feat_list.append(h)
            x = h

        h = torch.cat(feat_list, dim=-1)
        h = self.classifier(h)
        return h

    def reset_parameters(self):
        for i in range(self.info_dict['n_layers']):
            self.enc[i].reset_parameters()
            self.bns[i].reset_parameters()


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


class APPNPNet(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.enc = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=info_dict['dropout'])
        self.prop = APPNP(K=10, alpha=0.1, cached=True)
        for i in range(info_dict['n_layers']):
            in_dim = info_dict['in_dim'] if i == 0 else info_dict['hid_dim']
            out_dim = info_dict['out_dim'] if i == info_dict['n_layers'] - 1 else info_dict['hid_dim']
            bn = False if i == (info_dict['n_layers'] - 1) else info_dict['bn']
            self.enc.append(nn.Linear(in_dim, out_dim))
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())

    def forward(self, x, edge_index):
        # APPNP first step: predict
        for i in range(self.info_dict['n_layers']):
            x = self.dropout(x)
            h = self.enc[i](x)
            if i < self.info_dict['n_layers'] - 1:
                h = self.bns[i](h)
                h = self.act(h)
            x = h
        # APPNP second step: propagate
        h = self.prop(x, edge_index)
        x = h
        return x

    def reset_parameters(self):
        for i in range(self.info_dict['n_layers']):
            self.enc[i].reset_parameters()
            self.bns[i].reset_parameters()


class GCN2(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.enc = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=info_dict['dropout'])
        for i in range(info_dict['n_layers'] + 2):
            in_dim = info_dict['in_dim'] if i == 0 else info_dict['hid_dim']
            out_dim = info_dict['out_dim'] if i == info_dict['n_layers'] + 1 else info_dict['hid_dim']
            bn = False if i == (info_dict['n_layers'] - 1) else info_dict['bn']
            if (i == 0) or (i == info_dict['n_layers'] + 1):
                self.enc.append(nn.Linear(in_dim, out_dim))  # the input/ output transform layer
            else:
                self.enc.append(GCN2Conv(out_dim, alpha=0.1, theta=0.5, layer=i, shared_weights=True, normalize=False))
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())

    def forward(self, x, edge_index):
        x = self.dropout(x)
        # input transform
        x = x_0 = self.act(self.bns[0](self.enc[0](x)))
        edge_index = gcn_norm(edge_index)
        for i in range(1, self.info_dict['n_layers'] + 1):
            x = self.dropout(x)
            h = self.enc[i](x, x_0, edge_index=edge_index[0], edge_weight=edge_index[1])
            h = self.bns[i](h)
            h = self.act(h)
            x = h

        # output transform
        x = self.enc[-1](x)
        return x

    def reset_parameters(self):
        for i in range(self.info_dict['n_layers']):
            self.enc[i].reset_parameters()
            self.bns[i].reset_parameters()


class GIN(nn.Module):
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
            self.enc.append(GINConv(nn.Linear(in_dim, out_dim), train_eps=True))
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
