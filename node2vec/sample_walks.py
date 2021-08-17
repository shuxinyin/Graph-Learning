import numpy as np
import random
import pgl

def random_walk(g, nodes, max_depth):
    walk_paths = []
    # init
    for node in nodes:
        walk_paths.append([node])

    cur_walk_ids = np.arange(0, len(nodes))
    cur_nodes = np.array(nodes)
    for l in range(max_depth - 1):
        # select the walks not end
        cur_succs = g.successor(cur_nodes)
        mask = [len(succ) > 0 for succ in cur_succs]

        if np.any(mask):
            cur_walk_ids = cur_walk_ids[mask]
            cur_nodes = cur_nodes[mask]
            cur_succs = cur_succs[mask]
        else:
            # stop when all nodes have no successor
            break

        outdegree = [len(cur_succ) for cur_succ in cur_succs]
        sample_index = np.floor(
            np.random.rand(cur_succs.shape[0]) * outdegree).astype("int64")

        nxt_cur_nodes = []
        for s, ind, walk_id in zip(cur_succs, sample_index, cur_walk_ids):
            walk_paths[walk_id].append(s[ind])
            nxt_cur_nodes.append(s[ind])
        cur_nodes = np.array(nxt_cur_nodes)
    return walk_paths


def node2vec_walk(graph, nodes, max_depth, p=1.0, q=1.0):
    if p == 1.0 and q == 1.0:
        return random_walk(graph, nodes, max_depth)

    walk = []
    # init
    for node in nodes:
        walk.append([node])

    cur_walk_ids = np.arange(0, len(nodes))
    cur_nodes = np.array(nodes)
    prev_nodes = np.array([-1] * len(nodes), dtype="int64")
    prev_succs = np.array([[]] * len(nodes), dtype="int64")
    for l in range(max_depth):
        # select the walks not end
        cur_succs = graph.successor(cur_nodes)

        mask = [len(succ) > 0 for succ in cur_succs]
        if np.any(mask):
            cur_walk_ids = cur_walk_ids[mask]
            cur_nodes = cur_nodes[mask]
            prev_nodes = prev_nodes[mask]
            prev_succs = prev_succs[mask]
            cur_succs = cur_succs[mask]
        else:
            # stop when all nodes have no successor
            break
        num_nodes = cur_nodes.shape[0]
        nxt_nodes = np.zeros(num_nodes, dtype="int64")

        for idx, (
                succ, prev_succ, walk_id, prev_node
        ) in enumerate(zip(cur_succs, prev_succs, cur_walk_ids, prev_nodes)):
            sampled_succ = node2vec_sample(succ, prev_succ,
                                                        prev_node, p, q)
            walk[walk_id].append(sampled_succ)
            nxt_nodes[idx] = sampled_succ

        prev_nodes, prev_succs = cur_nodes, cur_succs
        cur_nodes = nxt_nodes
    return walk


def node2vec_walk(graph, nodes, max_depth, p=1.0, q=1.0):
    if p == 1.0 and q == 1.0:
        return random_walk(graph, nodes, max_depth)

    walk = []
    # init
    for node in nodes:
        walk.append([node])

    cur_walk_ids = np.arange(0, len(nodes))
    cur_nodes = np.array(nodes)
    prev_nodes = np.array([-1] * len(nodes), dtype="int64")
    prev_succs = np.array([[]] * len(nodes), dtype="int64")
    for l in range(max_depth):
        # select the walks not end
        cur_succs = graph.successor(cur_nodes) # all the successor

        mask = [len(succ) > 0 for succ in cur_succs]
        if np.any(mask):
            cur_walk_ids = cur_walk_ids[mask]
            cur_nodes = cur_nodes[mask]
            prev_nodes = prev_nodes[mask]
            prev_succs = prev_succs[mask]
            cur_succs = cur_succs[mask]
        else:
            # stop when all nodes have no successor
            break
        num_nodes = cur_nodes.shape[0]
        nxt_nodes = np.zeros(num_nodes, dtype="int64")
        print(l, cur_succs, prev_succs, cur_walk_ids, prev_nodes)
        for idx, (
                succ, prev_succ, walk_id, prev_node
        ) in enumerate(zip(cur_succs, prev_succs, cur_walk_ids, prev_nodes)):
            sampled_succ = node2vec_sample(succ, prev_succ,
                                           prev_node, p, q)
            walk[walk_id].append(sampled_succ)
            nxt_nodes[idx] = sampled_succ

        prev_nodes, prev_succs = cur_nodes, cur_succs
        cur_nodes = nxt_nodes
    return walk


def node2vec_sample(succ, prev_succ, prev_node, p, q):
    """Fast implement of node2vec sampling
    """
    print("succ", succ, "prev_succ", prev_succ, "prev_node", prev_node)
    succ_len = len(succ)
    prev_succ_len = len(prev_succ)

    probs = list()
    prob_sum = 0

    prev_succ_set = list()
    for i in range(prev_succ_len):
        prev_succ_set.insert(0, prev_succ[i])

    for i in range(succ_len):
        if succ[i] == prev_node:
            prob = 1. / p
        elif len(prev_succ_set) > 0 and succ[i] != prev_succ_set[-1]:
            prob = 1.
        else:
            prob = 1. / q
        probs.append(prob)
        prob_sum += prob

    rand_num = random.uniform(0, 1) * prob_sum

    for i in range(succ_len):
        rand_num -= probs[i]
        if rand_num <= 0:
            sample_succ = succ[i]
            return sample_succ


if __name__ == "__main__":
    import dgl
    import torch

    # g = dgl.graph(([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]))
    # g.edata['weight'] = torch.FloatTensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.5])
    # sg = dgl.sampling.select_topk(g, k=1, nodes=[0], weight='weight', edge_dir="in")
    # sg2 = dgl.sampling.sample_neighbors(g, [0, 1], 1, prob='weight', edge_dir="out")
    # print(sg2.edges(order='eid'))
    #
    # cur_nodes = np.array([0, 1, 2])
    # cur_succs = dgl.sampling.sample_neighbors(g, cur_nodes, 1, edge_dir="out")
    # print(cur_succs.edges()[1].tolist())
    # mask = [1 for succ in cur_succs.edges()[1].tolist() if succ]

    cur_nodes = np.array([0])
    g1 = dgl.graph(([0, 1, 1, 2, 3, 3, 4], [1, 2, 3, 0, 0, 4, 2]))
    print(dgl.sampling.sample_neighbors(g1, cur_nodes, 4, edge_dir="out"))
    print(node2vec_walk(g1, cur_nodes, max_depth=3, p=4,  q=0.25))
