import time
import os
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import dgl

from A_graph_learning.deepwalk.model import SkipGramModel
from A_graph_learning.deepwalk.build_graph import Build_Graph
from A_graph_learning.deepwalk.dataset import NodesDataset
from A_graph_learning.deepwalk.utils import skip_gram_gen_pairs


class Call_Func():
    def __init__(self, g, half_win_size, walk_length=4, p=1, q=1):
        self.g = g
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.half_win_size = half_win_size

    def __call__(self, nodes):
        batch_src, batch_dst = list(), list()

        walks_list = list()
        walks = dgl.sampling.node2vec_random_walk(self.g, nodes, p=1, q=1, walk_length=self.walk_length)
        walks_list += walks.tolist()
        for walk in walks_list:
            src, dst = skip_gram_gen_pairs(walk, self.half_win_size)
            batch_src += src
            batch_dst += dst

        # shuffle pair
        batch_tmp = list(zip(batch_src, batch_dst))
        random.shuffle(batch_tmp)
        batch_src, batch_dst = zip(*batch_tmp)

        batch_src = torch.from_numpy(np.array(batch_src))
        batch_dst = torch.from_numpy(np.array(batch_dst))
        return batch_src, batch_dst


def main(config):
    print(torch.cuda.device_count(), torch.cuda.is_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    # device = torch.device(config.gpu)
    torch.backends.cudnn.benchmark = True

    GraphSet = Build_Graph(config.file_path, walk_length=config.walk_length, self_loop=True, undirected=True)
    graph = GraphSet.graph

    model = SkipGramModel(graph.num_nodes(), embed_dim=config.embed_dim)
    model.cuda()
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=float(config.lr))

    nodes_dataset = NodesDataset(graph.nodes())
    pair_generate_func = Call_Func(graph, config.half_win_size, config.walk_length, config.p, config.q)

    pair_loader = DataLoader(nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
                             collate_fn=pair_generate_func)

    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()

        loss_total = list()
        top_loss = 0
        tqdm_bar = tqdm(pair_loader, desc="Training epoch{epoch}".format(epoch=epoch))
        for i, (batch_src, batch_dst) in enumerate(tqdm_bar):
            batch_src = batch_src.cuda().long()
            batch_dst = batch_dst.cuda().long()

            batch_neg = np.random.randint(0, graph.num_nodes(), size=(batch_src.shape[0], config.neg_num))
            batch_neg = torch.from_numpy(batch_neg).cuda().long()  # change multi neg_num

            model.zero_grad()
            loss = model.forward(batch_src, batch_dst, batch_neg)
            loss.backward()
            optimizer.step()
            loss_total.append(loss.detach().item())

        if top_loss > np.mean(loss_total):
            top_loss = np.mean(loss_total)
            torch.save(model.state_dict(), config.save_path)
            print("Epoch: %03d; loss = %.4f saved path: %s" % (epoch, top_loss, config.save_path))
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))


if __name__ == "__main__":
    class ConfigClass():
        def __init__(self):
            self.lr = 0.005
            self.gpu = "0"
            self.epochs = 32
            self.embed_dim = 64
            self.batch_size = 10

            self.p = 1
            self.q = 1
            self.walk_num_per_node = 6
            self.walk_length = 12
            self.win_size = 6
            self.neg_num = 5
            self.save_path = "../out/blog_deepwalk_ckpt"
            self.file_path = "../data/blog/"


    config = ConfigClass()
    # main(config)

    # import argparse
    # from utils import load_config
    #
    # parser = argparse.ArgumentParser(description='bert classification')
    # parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    # args = parser.parse_args()
    # config = load_config(args.config)
