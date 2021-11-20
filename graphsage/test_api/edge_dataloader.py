import dgl

import torch

src = torch.tensor([1, 3, 5, 7, 9])
dst = torch.tensor([2, 4, 6, 8, 10])
g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])))

E = len(src)
reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])

train_eid = src
sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
dataloader = dgl.dataloading.EdgeDataLoader(
    g, train_eid, sampler, exclude='reverse_id',
    reverse_eids=reverse_eids,
    batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
for input_nodes, pair_graph, blocks in dataloader:
    print(input_nodes)
    print(pair_graph)
    # print(neg_graph)
    print(blocks)
