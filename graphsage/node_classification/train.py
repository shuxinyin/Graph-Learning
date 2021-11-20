import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import IterableDataset, Dataset, DataLoader

from dataloader import build_cora_dataset, HomoNodesSet, NodesGraphCollactor
from model import SAGENet
from sklearn.metrics import f1_score

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[seeds].to(device)
    batch_labels = labels[input_nodes].to(device)
    return batch_inputs, batch_labels


def evaluation(features, labels, test_mask,
               model, test_data_loader, loss_fcn, device='cpu'):
    model.eval()
    with torch.no_grad():
        acc_cnt = 0
        for step, (input_nodes, seeds, blocks) in enumerate(test_data_loader):
            batch_feat, batch_labels = load_subtensor(features, labels, seeds, input_nodes, device='cpu')
            blocks = [b.to(device) for b in blocks]
            batch_feat = batch_feat.to(device)
            batch_labels = batch_labels.to(device)
            bacth_pred = model(blocks, batch_feat)
            loss = loss_fcn(bacth_pred, batch_labels)
            batch_acc_cnt = (torch.argmax(bacth_pred, dim=1) == batch_labels.long()).float().sum()
            acc_cnt += int(batch_acc_cnt)
            f1 = f1_score(batch_labels.detach().cpu(), torch.argmax(bacth_pred, dim=1).detach().cpu(), average='macro')
        print(f"Test: Loss:{loss}, cnt:{acc_cnt}, {torch.nonzero(test_mask).shape[0]},"
              f"Acc:{int(acc_cnt) / torch.nonzero(test_mask).shape[0]}")
        return int(acc_cnt) / torch.nonzero(test_mask).shape[0], f1


def train(args, graph):
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    graph.to(device)

    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']
    val_mask = graph.ndata['val_mask']
    in_feats = features.shape[1]
    n_classes = 7

    collator = NodesGraphCollactor(graph, neighbors_every_layer=args.neighbors_every_layer)

    batch_sampler = HomoNodesSet(graph, train_mask)
    data_loader = DataLoader(
        batch_sampler,
        batch_size=512,
        shuffle=True,
        num_workers=6,
        collate_fn=collator
    )

    test_batch_sampler = HomoNodesSet(graph, test_mask)
    test_data_loader = DataLoader(
        test_batch_sampler,
        batch_size=1000,
        shuffle=False,
        num_workers=6,
        collate_fn=collator
    )

    # Define model and optimizer
    model = SAGENet(in_feats, args.num_hidden, n_classes,
                    args.num_layers, F.relu, args.dropout)
    model.cuda()

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                            weight_decay=0.1, amsgrad=False)
    top_acc, top_f1 = 0, 0
    for epoch in range(args.num_epochs):
        acc_cnt = 0
        for step, (input_nodes, seeds, blocks) in enumerate(data_loader):
            batch_feat, batch_labels = load_subtensor(features, labels, seeds, input_nodes, device=device)
            blocks = [b.to(device) for b in blocks]
            batch_feat = batch_feat.to(device)
            batch_labels = batch_labels.to(device)
            bacth_pred = model(blocks, batch_feat)
            loss = loss_fcn(bacth_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_acc_cnt = (torch.argmax(bacth_pred, dim=1) == batch_labels.long()).float().sum()
            acc_cnt += int(batch_acc_cnt)
        print(f"Train Epoch:{epoch}, Loss:{loss}, Acc:{int(acc_cnt) / torch.nonzero(train_mask).shape[0]}")

        # evaluation()

        acc, f1 = evaluation(features, labels, test_mask, model, test_data_loader, loss_fcn, device)
        if top_f1 < f1:
            top_acc, top_f1 = acc, f1
    print(f"Test Top Acc: {top_acc}, F1:{top_f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameter set")
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--neighbors_every_layer', type=list, default=[10], help="or [10, 5]")
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument("--gpu", type=str, default='0',
                        help="gpu or cpu")
    args = parser.parse_args()
    graph = build_cora_dataset(add_symmetric_edges=True, add_self_loop=True)

    train(args, graph)
