import time
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from model import SkipGramModel
from build_graph import Build_Graph
from dataset import NodesDataset, Collate_Func


def main(config):
    print(torch.cuda.device_count(), torch.cuda.is_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    # device = torch.device(config.gpu)
    torch.backends.cudnn.benchmark = True

    GraphSet = Build_Graph(config.file_path, walk_length=5, self_loop=True, undirected=True)
    graph = GraphSet.graph

    model = SkipGramModel(graph.num_nodes(), embed_dim=config.embed_dim)
    model.cuda()
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=float(config.lr))

    nodes_dataset = NodesDataset(graph.nodes())
    pair_generate_func = Collate_Func(graph, config)

    pair_loader = DataLoader(nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
                             collate_fn=pair_generate_func)

    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()

        loss_total = list()
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

        torch.save(model.state_dict(), config.save_path)
        # torch.save(model.state_dict(), config.save_path)
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))


if __name__ == "__main__":
    class ConfigClass():
        def __init__(self, lr=0.05, gpu="0"):
            self.lr = 0.05
            self.gpu = "0"
            self.epochs = 200
            self.embed_dim = 32
            self.batch_size = 32
            self.walk_num_per_node = 4
            self.walk_length = 12
            self.win_size = 5
            self.neg_num = 3
            self.save_path = "../out/blog_deepwalk_ckpt"
            self.file_path = "../data/blog/"

    config = ConfigClass()
    main(config)

    # import argparse
    # from utils import load_config
    #
    # parser = argparse.ArgumentParser(description='bert classification')
    # parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    # args = parser.parse_args()
    # config = load_config(args.config)

