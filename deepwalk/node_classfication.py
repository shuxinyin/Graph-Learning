import time
import os
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics
from sklearn.model_selection import train_test_split

from model import NodeClassification


class NodesDataset(Dataset):
    def __init__(self, nodes, labels):
        self.nodes = nodes
        self.labels = labels

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, index):
        return self.nodes[index], self.labels[index]


def evaluate(test_nodes_loader, model, loss_func):
    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for i, (batch_nodes, batch_labels) in enumerate(test_nodes_loader):
        batch_nodes = batch_nodes.cuda()
        batch_labels = batch_labels.cuda()

        out = model(batch_nodes)
        loss = loss_func(out, batch_labels)
        total_loss += loss.detach().item()

        label = batch_labels.data.cpu().numpy()
        predic = torch.max(out.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, label)
        predict_all = np.append(predict_all, predic)

    f1_score = metrics.f1_score(labels_all, predict_all, average="macro")
    return f1_score, total_loss / len(test_nodes_loader)


def get_map_dict(dict_path):
    a_file = open(dict_path, "rb")
    map_dict = pickle.load(a_file)
    return map_dict


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    # device = torch.device(config.gpu)
    torch.backends.cudnn.benchmark = True

    param = torch.load(config.save_path)
    emb = param["embed_nodes.weight"]
    print(emb.shape)
    node_map_dict = get_map_dict(config.file_path + "node_map_dic.pkl")

    df_label = pd.read_csv(config.file_path + "blog-label.txt", header=None, sep="\t", names=["nodes", "label"])
    df_label.label = df_label.label.map(lambda x: x - 1)
    df_label.nodes = df_label.nodes.map(node_map_dict)
    train, test = train_test_split(df_label, test_size=0.1)


    train_nodes_dataset = NodesDataset(train.nodes.to_list(), train.label.to_list())
    train_nodes_loader = DataLoader(train_nodes_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_nodes_dataset = NodesDataset(test.nodes.to_list(), test.label.to_list())
    test_nodes_loader = DataLoader(test_nodes_dataset, batch_size=256, shuffle=False, num_workers=4)
    print("--", len(train_nodes_loader), len(test_nodes_loader))

    model = NodeClassification(emb, num_class=config.num_class)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), betas=(0.9, 0.999), eps=1e-08,
                                  weight_decay=0.01, amsgrad=False)
    loss_func = F.cross_entropy

    start_time = time.time()
    for epoch in range(config.epochs):
        loss_total = list()
        for i, (batch_nodes, batch_labels) in enumerate(train_nodes_loader):
            batch_nodes = batch_nodes.cuda()
            batch_labels = batch_labels.cuda()

            model.zero_grad()
            prob = model(batch_nodes)
            loss = loss_func(prob, batch_labels)
            loss.backward()
            optimizer.step()
            loss_total.append(loss.detach().item())
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
        f1, loss = evaluate(test_nodes_loader, model, loss_func)
        print("Epoch: %03d; f1 = %.4f loss  %.4f" % (epoch, f1, loss))


if __name__ == "__main__":
    class ConfigClass():
        def __init__(self):
            self.lr = 0.05
            self.gpu = "0"
            self.epochs = 16
            self.num_class = 39
            self.save_path = "../out/blog_deepwalk_ckpt"
            self.file_path = "../data/blog/"

    config = ConfigClass()
    main(config)
