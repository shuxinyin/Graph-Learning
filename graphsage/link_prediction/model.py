import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn


class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, output_dims, act=F.relu, dropout=0.5, bias=True):
        super().__init__()

        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.Q = nn.Linear(input_dims, output_dims)
        self.W = nn.Linear(input_dims + output_dims, output_dims)
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dims))
        else:
            self.register_parameter('bias', None)
        # self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, g, h, weights=None):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        with g.local_scope():
            if weights:
                g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
                g.edata['w'] = weights.float()
                g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
                g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
                n = g.dstdata['n']
                ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
                z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
                z_norm = z.norm(2, 1, keepdim=True)
                z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
                z = z / z_norm
            else:
                g.srcdata['n'] = self.Q(h_src)
                g.update_all(fn.copy_src('n', 'm'), fn.mean('m', 'neigh'))  # aggregation
                n = g.dstdata['neigh']
                z = self.act(self.W(torch.cat([n, h_dst], 1))) + self.bias
                z_norm = z.norm(2, 1, keepdim=True)
                z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
                z = z / z_norm
            return z


class SAGENet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dims,
                 n_layers, act=F.relu, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(WeightedSAGEConv(input_dim, hidden_dims, act, dropout))
        for _ in range(n_layers - 2):
            self.convs.append(WeightedSAGEConv(hidden_dims, hidden_dims,
                                               act, dropout))
        self.convs.append(WeightedSAGEConv(hidden_dims, output_dims,
                                           act, dropout))
        self.dropout = nn.Dropout(dropout)
        # self.act = act

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.convs, blocks)):
            h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]  # 这只取dst点，从下往上aggregate，得到头结点
            h = layer(block, (h, h_dst))
            if l != len(self.convs) - 1:
                h = self.dropout(h)
        return h



