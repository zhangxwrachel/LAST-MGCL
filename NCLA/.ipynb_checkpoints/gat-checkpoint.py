"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch.nn as nn
from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope):
        super(GAT, self).__init__()
        self.g = g               # 图数据
        self.num_layers = num_layers # GAT 层数
        self.num_hidden = num_hidden  # 隐藏层维度
        self.gat_layers = nn.ModuleList()# 存储 GATConv 层的列表
        self.activation = activation# 激活函数
        print('GATConv层输入：',in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation)
        print('num_layers:',num_layers)

        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            print('heads[l - 1]:',heads[l - 1],'heads[l]:',heads[l])
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))

    def forward(self, inputs):
        heads = []
        h = inputs
        # get hidden_representation
        for l in range(self.num_layers):
            print('NCLA当中h的维度：',h.shape)
            temp = h.flatten(1)
            print('NCLA当中temp的维度：',temp.shape)
            print('NCLA当中temp的维度：',temp.shape,type(temp))
            print("Number of nodes:", self.g.num_nodes())
            print("Number of edges:", self.g.num_edges())
            h =self.gat_layers[l](self.g, temp)
            print('NCLA当中gat_layers后h的维度：',h.shape)
        # get heads
        for i in range(h.shape[1]):
            heads.append(h[:, i])
        return heads
