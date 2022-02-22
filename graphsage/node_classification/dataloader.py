import argparse
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader

import dgl
from dgl.data import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset, CoraGraphDataset

import random

random.seed(123)


class HomoNodesSet(Dataset):
    def __init__(self, g, mask):
        # only load masked node for training/testing
        self.g = g
        self.nodes = g.nodes()[mask].tolist()

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, index):
        heads = self.nodes[index]
        return heads


class NodesGraphCollactor(object):
    """
    select heads/tails/neg_tails's neighbors for aggregation
    """

    def __init__(self, g, neighbors_every_layer=[5, 1]):
        self.g = g
        self.neighbors_every_layer = neighbors_every_layer

    def __call__(self, batch):
        blocks, seeds = self.sample_blocks(batch)
        return batch, seeds, blocks

    def sample_blocks(self, seeds):
        blocks = []
        for n_neighbors in self.neighbors_every_layer:
            frontier = dgl.sampling.sample_neighbors(
                self.g,
                seeds,
                fanout=n_neighbors,
                edge_dir='in')
            block = self.compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]  # 这里应该返回这一层的src node

            blocks.insert(0, block)
        return blocks, seeds

    def compact_and_copy(self, frontier, seeds):
        # 将第一轮的dst节点与frontier压缩成block
        # 并设置block的seeds 为 output nodes，其他为input nodes
        block = dgl.to_block(frontier, seeds)
        for col, data in frontier.edata.items():
            if col == dgl.EID:
                continue
            block.edata[col] = data[block.edata[dgl.EID]]
        return block


def build_graph(args):
    # load graph news
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    elif args.dataset == 'cora':
        dataset = AMDataset()
    else:
        raise ValueError()

    g = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop('train_mask')
    test_mask = g.nodes[category].data.pop('test_mask')
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop('labels')
    print(g)
    print(len(g.etypes), g.etypes)
    return g


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
    # graph = build_graph(args)
    graph = build_cora_dataset()
    train_mask = graph.ndata['train_mask']

    batch_sampler = HomoNodesSet(graph, train_mask)
    collator = NodesGraphCollactor(graph, neighbors_every_layer=[5, 2])
    dataloader = DataLoader(
        batch_sampler,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        collate_fn=collator
    )

    for step, (seed, blocks_nodes, blocks) in enumerate(dataloader):
        print("------------------------")
        print(seed)
        print(blocks_nodes)
        print(blocks)
        for b in blocks:
            print(b.edges())
        break
