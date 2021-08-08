import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np


# from gensim.models import Word2Vec


class SkipGramModel(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super(SkipGramModel, self).__init__()
        self.num_nodes = num_nodes
        self.emb_dimension = embed_dim

        self.embed_nodes = nn.Embedding(self.num_nodes, self.emb_dimension, sparse=True)
        nn.init.xavier_uniform_(self.embed_nodes.weight)
        # initialze embedding
        # initrange = 1.0 / self.emb_dimension
        # nn.init.uniform_(self.embed_nodes.weight.data, -initrange, initrange)

    def forward(self, src, pos, neg):
        embed_src = self.embed_nodes(src)  # (B, d)
        embed_pos = self.embed_nodes(pos)  # (B, d)
        embed_neg = self.embed_nodes(neg)  # (B, neg_num, d)
        # print(embed_src.shape, embed_pos.shape, embed_neg.shape)

        pos_socre = torch.sum(torch.matmul(embed_src, embed_pos.transpose(0, 1)), 1)
        pos_socre = -F.logsigmoid(pos_socre)

        neg_socre = torch.sum(torch.matmul(embed_src, embed_neg.transpose(1, 2)), (1, 2))
        neg_socre = -F.logsigmoid(-neg_socre)
        # print(pos_socre.shape, neg_socre.shape)

        loss = torch.mean(pos_socre + neg_socre)
        return loss


class NodeClassification(nn.Module):
    def __init__(self, emb, num_class=38):
        super(NodeClassification, self).__init__()
        self.emb = emb
        self.num_class = num_class
        self.fc = nn.Linear(self.emb.shape[1], self.num_class)

    def forward(self, node):
        node_emb = self.emb[node]
        prob = self.fc(node_emb)

        return prob


def skip_gram_model_test():
    model = SkipGramModel(1000, embed_dim=32)
    model.cuda()

    src = np.random.randint(0, 100, size=10)
    src = torch.from_numpy(src).cuda().long()

    dst = np.random.randint(0, 100, size=10)
    dst = torch.from_numpy(dst).cuda().long()

    neg = np.random.randint(0, 100, size=(10, 5))
    neg = torch.from_numpy(neg).cuda().long()

    print(src.shape, dst.shape, neg.shape)

    print(model(src, dst, neg))


def node_classfication_model_test():
    emb = torch.randn(10, 4)
    print(emb)
    model = NodeClassification(emb)

    nodes = torch.tensor([1, 2, 3])
    print(nodes)
    print(emb[nodes])
    print(model(nodes))


if __name__ == "__main__":
    node_classfication_model_test()
