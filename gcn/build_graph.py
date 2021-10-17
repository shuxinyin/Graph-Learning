import math
import random
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np

import dgl
import torch

from utils import preprocess_adj
from dgl.data import CoraGraphDataset


class GraphBuild(object):
    def __init__(self):
        # self.graph = self.build_graph_test()
        self.graph = self.build_graph_cora()
        self.adj = self.get_adj(self.graph)
        self.features = self.init_node_feat(self.graph)

    def build_graph_test(self):
        """a demo graph: just for graph test
        """
        src_nodes = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6])
        dst_nodes = torch.tensor([1, 2, 0, 2, 0, 1, 3, 4, 5, 6, 2, 3, 3, 3])
        graph = dgl.graph((src_nodes, dst_nodes))
        # edges weights if edges has else 1
        graph.edata["w"] = torch.ones(graph.num_edges())
        return graph

    def build_graph_cora(self):
        # Default: ~/.dgl/
        data = CoraGraphDataset()
        graph = data[0]

        return graph

    def convert_symmetric(self, X, sparse=True):
        # add symmetric edges
        if sparse:
            X += X.T - sp.diags(X.diagonal())
        else:
            X += X.T - np.diag(X.diagonal())
        return X

    def add_self_loop(self, graph):
        # add self loop
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        return graph

    def get_adj(self, graph):
        graph = self.add_self_loop(graph)
        # edges weights if edges has weights else 1
        graph.edata["w"] = torch.ones(graph.num_edges())
        adj = coo_matrix((graph.edata["w"], (graph.edges()[0], graph.edges()[1])),
                         shape=(graph.num_nodes(), graph.num_nodes()))

        #  add symmetric edges
        adj = self.convert_symmetric(adj, sparse=True)
        # adj normalize and transform matrix to torch tensor type
        adj = preprocess_adj(adj, is_sparse=True)

        return adj

    def init_node_feat(self, graph):
        # init graph node features
        self.nfeat_dim = graph.number_of_nodes()

        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))
        indices = torch.from_numpy(
            np.vstack((row, col)).astype(np.int64))
        values = torch.ones(self.nfeat_dim)

        features = torch.sparse.FloatTensor(indices, values,
                                            (self.nfeat_dim, self.nfeat_dim))
        return features


if __name__ == "__main__":
    GraphSet = GraphBuild()
    graph = GraphSet.graph
    graph = GraphSet.add_self_loop(graph)
    print(graph)
    # print(graph.ndata['feat'].shape)
    features = GraphSet.init_node_feat(graph)  # (num_nodes, num_nodes)
    adj = GraphSet.get_adj(graph)
    print(features.shape, adj.shape)
    print(adj.shape)  # (10556, 10556)
    print(graph.edges())
