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
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, src, pos, neg):
        embed_src = self.embed_nodes(src)  # (B, d)
        embed_pos = self.embed_nodes(pos)  # (B, d)
        embed_neg = self.embed_nodes(neg)  # (B, neg_num, d)
        # print(embed_src.shape, embed_pos.shape, embed_neg.shape)

        pos_logits = torch.matmul(embed_src, embed_pos.transpose(0, 1))
        ones_label = torch.ones_like(pos_logits)
        # print(pos_logits.shape, ones_label.shape)
        pos_loss = self.loss(pos_logits, ones_label)

        neg_logits = torch.matmul(embed_src, embed_neg.transpose(1, 2))
        zeros_label = torch.zeros_like(neg_logits)
        # print(neg_logits.shape, zeros_label.shape)
        neg_loss = self.loss(neg_logits, zeros_label)

        loss = (pos_loss + neg_loss) / 2
        return loss


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


if __name__ == "__main__":
    skip_gram_model_test()
