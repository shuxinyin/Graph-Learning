import sys
import argparse
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader

import dgl
from dgl.data import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset, CoraGraphDataset

import random
from loguru import logger
random.seed(123)


class NodesSet(Dataset):
    def __init__(self, g, neg_num=1):
        # only load masked node for training/testing
        self.g = g
        self.nodes = g.nodes().tolist()
        self.neg_num = neg_num  # wait to complement

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, index):
        heads = self.nodes[index]
        pos_nodes = dgl.sampling.random_walk(self.g,
                                             heads,
                                             length=1)[0][:, 1].tolist()[0]

        neg_nodes = random.sample(self.nodes, k=self.neg_num)[0]
        # logger.info(f"heads: {heads}")
        # logger.info(f"pos_nodes: {pos_nodes}")
        # logger.info(f"neg_nodes: {neg_nodes}")

        return heads, pos_nodes, neg_nodes


class NodesGraphCollactor(object):
    """
    select heads/tails/neg_tails's neighbors for aggregation
    """

    def __init__(self, g, neighbors_every_layer=[5, 1]):
        self.g = g
        self.nodes = g.nodes().tolist()
        self.neighbors_every_layer = neighbors_every_layer


    def __call__(self, batch):
        # logger.info(f"batch: {batch}")
        # pos_nodes, neg_nodes = self.sample_pos_neg_nodes(batch)
        # heads: [2569, 741]
        # pos_nodes: tensor([2268, 1423])
        # neg_nodes: [[1827, 1051, 1862, 477, 1595], [1907, 634, 88, 495, 2697]]
        heads = [b[0] for b in batch]
        tails = [b[1] for b in batch]
        neg_tails = [b[2] for b in batch]

        heads, tails, neg_tails = torch.tensor(heads), torch.tensor(tails), torch.tensor(neg_tails)
        # logger.info(heads, tails, neg_tails)
        pos_graph, neg_graph, blocks, all_seeds = self.sample_from_item_pairs(heads, tails, neg_tails)

        return pos_graph, neg_graph, blocks, set(all_seeds)

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.number_of_nodes())
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.number_of_nodes())
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks, all_seeds = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks, all_seeds

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks, all_seeds = [], []
        for n_neighbors in self.neighbors_every_layer:
            frontier = dgl.sampling.sample_neighbors(
                self.g,
                seeds,
                fanout=n_neighbors,
                edge_dir='in')
            if heads is not None:
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
            block = self.compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]  # 这里应该返回这一层的src node
            # logger.info(f"seeds: {seeds}")
            all_seeds += seeds.tolist()
            blocks.insert(0, block)
        return blocks, all_seeds

    def compact_and_copy(self, frontier, seeds):
        # 将第一轮的dst节点与frontier压缩成block
        # 并设置block的seeds 为 output nodes，其他为input nodes
        block = dgl.to_block(frontier, seeds)
        for col, data in frontier.edata.items():
            if col == dgl.EID:
                continue
            block.edata[col] = data[block.edata[dgl.EID]]
        return block


def build_cora_dataset(add_symmetric_edges=True, add_self_loop=True):
    dataset = CoraGraphDataset()
    graph = dataset[0]

    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    labels = graph.ndata['label']
    feat = graph.ndata['feat']

    if add_symmetric_edges:
        edges = graph.edges()
        graph.add_edges(edges[1], edges[0])

    graph = dgl.remove_self_loop(graph)
    if add_self_loop:
        graph = dgl.add_self_loop(graph)
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameter set")
    parser.add_argument('--dataset', type=str, default='aifb')

    args = parser.parse_args()
    graph = build_cora_dataset()
    train_mask = graph.ndata['train_mask']

    batch_sampler = NodesSet(graph, train_mask)
    collator = NodesGraphCollactor(graph, neighbors_every_layer=[5, 2])
    dataloader = DataLoader(
        batch_sampler,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        collate_fn=collator
    )
    # for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
    for step, (batch, heads_seeds, heads_blocks, all_seeds) in enumerate(dataloader):
        logger.info(f"---------step: {step}")
        logger.info(f"nodes: {batch}")
        logger.info(f"seeds: {heads_seeds}")
        logger.info(f"graph: {heads_blocks}")
        logger.info(f"grapall_seedsh: {all_seeds}")

        break
