import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F

import dgl
from dgl import DGLGraph
from dgl.data import CoraGraphDataset

from model import GATModel


def load_cora_data(args):
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    else:
        data = None
        raise NotImplementedError

    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    # num_feats = features.shape[1]
    # n_classes = data.num_labels
    # n_edges = data.graph.number_of_edges()
    return g, features, labels, train_mask, val_mask, test_mask


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main(args):
    g, features, labels, train_mask, val_mask, test_mask = load_cora_data(args)

    if args.add_self_loop:
        # add self loop
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    model = GATModel(g,
                     in_dim=features.size()[1],
                     hidden_dim=8,
                     out_dim=7,
                     num_heads=8)
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    top_val_acc, top_test_acc = 0, 0
    cost_time = []
    for epoch in range(args.epochs):
        t0 = time.time()

        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, features, labels, val_mask)

        print("Epoch {:03d} | val acc {:.4f}|  Loss {:.4f} | Time(s) {:.8f}".format(
            epoch, val_acc, loss.item(), time.time() - t0))

        if top_val_acc <= val_acc:
            top_val_acc = val_acc
            acc = evaluate(model, features, labels, test_mask)
            top_test_acc = max(top_test_acc, acc)
            print("Test Accuracy {:.4f}".format(acc))
    print(f"Top Test Acc: {top_test_acc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--dataset", type=str, default='cora',
                        help="which dataset to use.")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--add_self_loop", type=bool, default=True,
                        help="add self loop")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    args = parser.parse_args()
    print(args)

    main(args)
