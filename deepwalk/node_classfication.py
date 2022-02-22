import time
import os
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics
from sklearn.model_selection import train_test_split


def get_map_dict(dict_path):
    a_file = open(dict_path, "rb")
    map_dict = pickle.load(a_file)
    return map_dict


def label_data(config, mode="train"):
    node_map_dict = get_map_dict(config.file_path + "node_map_dic.pkl")
    df = pd.read_csv(config.file_path + "blog-label.txt", header=None, sep="\t", names=["nodes", "label"])
    df.nodes = df.nodes.map(node_map_dict)

    df_label = pd.crosstab(df.nodes, df.label).gt(0).astype(int)
    df_label = df_label.reset_index()

    train, test = train_test_split(df_label, test_size=0.1, random_state=123, shuffle=True)
    if mode == "train":
        node = train.nodes.to_list()
        label = train.drop('nodes', axis=1).values.tolist()
    else:
        node = test.nodes.to_list()
        label = test.drop('nodes', axis=1).values.tolist()

    return np.array(node), np.array(label)



class NodesDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.nodes, self.labels = label_data(config, mode=mode)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, index):
        return self.nodes[index], self.labels[index]


class NodeClassification(nn.Module):
    def __init__(self, emb, num_class=38):
        super(NodeClassification, self).__init__()
        self.emb = nn.Embedding.from_pretrained(emb, freeze=True)
        self.size = emb.shape[1]
        self.num_class = num_class
        self.fc = nn.Linear(self.size, self.num_class)

    def forward(self, node):
        # node = torch.tensor(node).to(torch.int64)
        node_emb = self.emb(node)
        prob = self.fc(node_emb)

        return prob


def evaluate(test_nodes_loader, model):
    model.eval()

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for i, (batch_nodes, batch_labels) in enumerate(test_nodes_loader):
        batch_nodes = batch_nodes.cuda().long()
        batch_labels = batch_labels.cuda().float()

        logit = model(batch_nodes)
        probs = torch.sigmoid(logit)

        label = torch.max(batch_labels.data, 1)[1].cpu().numpy()
        pred = torch.max(probs.data, 1)[1].cpu().numpy()
        # predic = torch.max(probs.news, 1)[1].cpu().numpy()
        print(pred.size, label.size)

        labels_all = np.append(labels_all, label)
        predict_all = np.append(predict_all, pred)
    print(labels_all.size, predict_all.size)
    f1_score = metrics.f1_score(labels_all, predict_all, average="macro")
    return f1_score


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    # device = torch.device(config.gpu)
    torch.backends.cudnn.benchmark = True

    param = torch.load(config.save_path)
    emb = param["embed_nodes.weight"]
    print(emb.shape)

    train_nodes_dataset = NodesDataset(config, "train")
    train_nodes_loader = DataLoader(train_nodes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_nodes_dataset = NodesDataset(config, "test")
    test_nodes_loader = DataLoader(test_nodes_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    print("--", len(train_nodes_loader), len(test_nodes_loader))

    model = NodeClassification(emb, num_class=config.num_class)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), betas=(0.9, 0.999), eps=1e-08,
                                  weight_decay=0.01, amsgrad=False)
    loss_func = nn.BCEWithLogitsLoss()

    start_time = time.time()
    for epoch in range(config.epochs):
        loss_total = list()
        for i, (batch_nodes, batch_labels) in enumerate(train_nodes_loader):
            batch_nodes = batch_nodes.cuda().long()
            batch_labels = batch_labels.cuda().float()

            model.zero_grad()
            logit = model(batch_nodes)
            probs = torch.sigmoid(logit)
            loss = loss_func(probs, batch_labels)
            loss.backward()
            optimizer.step()

            loss_total.append(loss.detach().item())
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
        f1 = evaluate(test_nodes_loader, model)
        print("Epoch: %03d; f1 = %.4f" % (epoch, f1))


if __name__ == "__main__":
    class ConfigClass():
        def __init__(self):
            self.lr = 0.05
            self.gpu = "0"
            self.epochs = 32
            self.batch_size = 256
            self.num_class = 39
            self.save_path = "../out/blog_deepwalk_ckpt"
            self.file_path = "../data/blog/"


    config = ConfigClass()
    main(config)
