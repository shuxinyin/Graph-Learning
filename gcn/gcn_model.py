import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features_dim, out_features_dim, activation=None, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.activation = activation
        self.weight = Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.news.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            # self.bias.news.uniform_(-stdv, stdv)
            nn.init.zeros_(self.bias)

    def forward(self, infeatn, adj):
        '''
        infeatn: init feature(H)
        adj: A
        '''
        support = torch.spmm(infeatn, self.weight)  # H*W  # (in_feat_dim, in_feat_dim) * (in_feat_dim, out_dim)
        output = torch.spmm(adj, support)  # A*H*W  # (in_feat_dim, in_feat_dim) * (in_feat_dim, out_dim)
        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(Module):
    def __init__(self, nfeat, nhid, nclass, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConvolution(nfeat, nhid, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConvolution(nhid, nhid, activation=activation))
        # output layer
        self.layers.append(GraphConvolution(nhid, nclass))
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, adj):

        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, adj)
        return h
