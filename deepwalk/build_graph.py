import os
import copy
import numpy as np
import scipy.sparse as sp
import pickle
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import download, _get_dgl_url, get_download_dir, extract_archive
import random
import time
import dgl


def make_undirected(G):
    G.add_edges(G.edges()[1], G.edges()[0])
    return G


def find_connected_nodes(G):
    nodes = G.out_degrees().nonzero().squeeze(-1)
    return nodes


import time
import pandas as pd


class Build_Graph(object):
    def __init__(self, data_dir, walk_length=5, self_loop=True, undirected=True):
        self.edge_file_path = data_dir + "blog-net.txt"
        self.map_dict_save_path = data_dir + "node_map_dic.pkl"
        self.self_loop = self_loop
        self.undirected = undirected
        self.walk_length = walk_length

        self.edges, self.nodes, self.node2id, self.id2node = self.get_edges_and_mapdict(self.edge_file_path,
                                                                                        self_loop=self.self_loop,
                                                                                        undirected=self.undirected)
        self.save_dict(self.node2id, self.map_dict_save_path)

        self.graph = self.build_graph(self.edges)

        print("total nodes number: %d" % self.graph.num_nodes())
        print("total edges number: %d" % len(self.edges[0]))
    def get_edges_and_mapdict(self, file_path, self_loop=True, undirected=True):
        df_net = pd.read_csv(file_path, header=None, sep=" ", names=["src", "dst", "weight"])

        nodes = list(set(sorted(df_net.src.to_list() + df_net.dst.to_list())))
        node2id = dict(zip(nodes, range(len(nodes))))
        id2node = dict(zip(range(len(nodes)), nodes))

        src = df_net.src.map(node2id).to_list()
        dst = df_net.dst.map(node2id).to_list()

        if undirected:
            tmp = copy.deepcopy(src)
            src.extend(dst)
            dst.extend(tmp)

        if self_loop:
            src.extend(nodes)
            dst.extend(nodes)

        assert max(node2id.values()) == len(nodes) - 1, "error reading net, quit"

        return (src, dst), nodes, node2id, id2node

    def build_graph(self, edges):
        start = time.time()
        G = dgl.graph((torch.tensor(edges[0]), torch.tensor(edges[1])))
        t = time.time() - start
        print("Building DGLGraph in %.2fs" % t)
        return G

    def save_dict(self, map_dic, save_path):
        a_file = open(save_path, "wb")
        pickle.dump(map_dic, a_file)
        a_file.close()


if __name__ == "__main__":
    # net, node2id, id2node, sm = ReadTxtNet(file_path="youtube", undirected=True)
    file_path = "D:/Learn_Project/graph_work/data/blog/"
    GraphSet = Build_Graph(file_path, walk_length=5, self_loop=True, undirected=True)

    # Walk_Sampler = DeepwalkSampler(G, seeds, walk_length)
