import torch
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
        self.g = g
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))

    def forward(self, inputs):
        heads = []
        augmented_graph_features = []  # 存储每一层的增强图数据的特征
        augmented_graph_adj = []  # 存储每一层的增强图的邻接矩阵
        h = inputs
        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h = self.gat_layers[l](self.g, temp)
            augmented_graph_features.append(h)  # 将当前层的节点嵌入特征添加到列表中
            # adj = self.gat_layers[l].graph
            # augmented_graph_adj.append(adj)  # 将当前层的邻接矩阵添加到列表中
        # get heads
        for i in range(h.shape[1]):
            heads.append(h[:, i])
        return heads, augmented_graph_features
