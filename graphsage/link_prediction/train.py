import os
import argparse
from tqdm import tqdm
from loguru import logger
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import IterableDataset, Dataset, DataLoader

import dgl.function as fn
import sklearn.linear_model as lm
import sklearn.metrics as skm

from dataloader import build_cora_dataset, NodesSet, NodesGraphCollactor
from model import SAGENet


class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = torch.cat([pos_score, neg_score])
        label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss


def load_subtensor(nfeat, seeds, device='cpu'):
    """
    Extracts features and labels for a subset of nodes
    """
    # logger.info(len(seeds))
    seeds_feats = nfeat[list(seeds)].to(device)
    # batch_labels = labels[input_nodes].to(device)
    return seeds_feats


def compute_acc_unsupervised(emb, graph):
    """
    Compute the accuracy of prediction given the labels.
    """

    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']
    val_mask = graph.ndata['val_mask']
    train_nids = torch.LongTensor(np.nonzero(train_mask)).squeeze().cpu().numpy()
    val_nids = torch.LongTensor(np.nonzero(val_mask)).squeeze().cpu().numpy()
    test_nids = torch.LongTensor(np.nonzero(test_mask)).squeeze().cpu().numpy()

    emb = emb.cpu().detach().numpy()
    labels = graph.ndata['label'].cpu().numpy()
    train_labels = labels[train_nids]
    val_labels = labels[val_nids]
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=1000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_train = skm.f1_score(train_labels, pred[train_nids], average='micro')
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_train, f1_micro_eval, f1_micro_test


def train(args, graph):
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    graph.to(device)

    features = graph.ndata['feat']
    in_feats = features.shape[1]
    n_classes = 7

    collator = NodesGraphCollactor(graph, neighbors_every_layer=args.neighbors_every_layer)

    batch_sampler = NodesSet(graph)
    data_loader = DataLoader(
        batch_sampler,
        batch_size=512,
        shuffle=True,
        num_workers=6,
        collate_fn=collator
    )

    # should aggregate while testing.
    test_collator = NodesGraphCollactor(graph, neighbors_every_layer=[10000])
    test_data_loader = DataLoader(
        batch_sampler,
        batch_size=10000,
        shuffle=False,
        num_workers=6,
        collate_fn=test_collator
    )

    # Define model and optimizer
    model = SAGENet(in_feats, args.num_hidden, n_classes,
                    args.num_layers, F.relu, args.dropout)
    model.cuda()

    # loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                            weight_decay=0.1, amsgrad=False)
    top_acc, top_f1 = 0, 0
    for epoch in range(args.num_epochs):
        acc_cnt = 0
        for step, (pos_graph, neg_graph, blocks, all_seeds) in enumerate(data_loader):
            # pos_nodes, neg_nodes_batch = collator.sample_pos_neg_nodes(batch)
            # logger.info(len(batch), len(all_seeds))
            # logger.info(len(pos_nodes), len(neg_nodes_batch), torch.tensor(neg_nodes_batch).shape)
            feats = load_subtensor(features, all_seeds, device=device)
            # pos_feats = load_subtensor(features, pos_nodes, device=device)
            # neg_feats = load_subtensor(features, neg_nodes_batch, device=device)
            # logger.info(heads_feats.shape, pos_feats.shape, neg_feats.shape)
            blocks = [b.to(device) for b in blocks]
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            bacth_pred = model(blocks, feats)
            loss = loss_fcn(bacth_pred, pos_graph, neg_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # batch_acc_cnt = (torch.argmax(bacth_pred, dim=1) == batch_labels.long()).float().sum()
            # acc_cnt += int(batch_acc_cnt)
        logger.info(f"Train Epoch:{epoch}, Loss:{loss}")

        # evaluation
        for step, (pos_graph, neg_graph, blocks, all_seeds) in enumerate(test_data_loader):
            feats = load_subtensor(features, all_seeds, device=device)
            blocks = [b.to(device) for b in blocks]
            bacth_pred = model(blocks, feats)

        f1_micro_train, f1_micro_eval, f1_micro_test = compute_acc_unsupervised(bacth_pred, graph)
        if top_f1 < f1_micro_test:
            top_f1 = f1_micro_test
        logger.info(
            f" train f1:{f1_micro_train}， Val micro F1: {f1_micro_eval}, Test micro F1:{f1_micro_test}, TOP micro F1:{top_f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameter set")
    parser.add_argument('--num_epochs', type=int, default=64)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_layers', type=int, default=2)
    # TODO: multiple negative nodes
    parser.add_argument('--neighbors_every_layer', type=list, default=[10], help="or [10, 5]")
    parser.add_argument("--gpu", type=str, default='0',
                        help="gpu or cpu")
    args = parser.parse_args()
    graph = build_cora_dataset(add_symmetric_edges=True, add_self_loop=True)

    train(args, graph)
