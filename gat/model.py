import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats,
                 feat_drop=0.6, attn_drop=0.6,
                 negative_slope=0.2, residual=False, activation=None):
        ''' define a GAT layer:
            you can adjust the parameter of drop rate and negative_slope to get bette result
            for cora dataset because cora is too small to get overfitting.
        '''
        super(GATLayer, self).__init__()
        self.g = g
        self.dropout_feat = nn.Dropout(feat_drop)
        self.fc = nn.Linear(in_feats, out_feats, bias=False)
        self.dropout_attn = nn.Dropout(attn_drop)
        self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)
        self.activation = activation
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        src_e = self.attention_func(concat_z)
        src_e = self.leaky_relu(src_e)
        return {'e': src_e}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = self.dropout_attn(alpha)  # add attention dropout
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        h = self.dropout_feat(h)  # add feat dropout
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GATModel(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GATModel, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # input dimension: hidden_dim * num_heads
        # output dimension: hidden_dim * 1  (由于concat(num_heads)， 只输出一个头。)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h
