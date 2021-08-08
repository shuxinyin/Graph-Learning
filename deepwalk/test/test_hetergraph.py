import dgl
import torch


g = dgl.heterograph({('user', 'plays', 'game'): (torch.tensor([0]), torch.tensor([1])),
                     ('developer', 'develops', 'game'): (torch.tensor([1]), torch.tensor([2]))})

g.dstnodes('game')

g.dstnodes['game'].data['h'] = torch.ones(3, 1)
print(g.dstnodes['game'].data['h'])


g = dgl.heterograph({('user', 'follows', 'user'): (torch.tensor([0]), torch.tensor([1])),
                     ('developer', 'develops', 'game'): (torch.tensor([1]), torch.tensor([2]))})


g.dstnodes('developer')

g.dstnodes['developer'].data['h'] = torch.ones(2, 1)
print(g.dstnodes['developer'].data['h'])