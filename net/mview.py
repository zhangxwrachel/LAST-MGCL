import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class MLP(nn.Module):
    '''对应论文中的p和predictor'''

    def __init__(self, inp_size, outp_size, hidden_size):
        '''
        Args:

        inp_size: 输入的维度
        outp_size: 输出的维度
        hidden_size: 隐藏层的维度
        '''
        super().__init__()
        # MLP的架构是一个输入层，接一个隐藏层，再接PReLU进行非线性化处理，最后
        # 全连接层进行输出
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, outp_size)
        )

    def forward(self, x):
        '''数据通过MLP时没有进行其他的处理，因此仅仅通过net()'''
        return self.net(x)


class GraphEncoder(nn.Module):
    '''encoder包含gnn(论文中的f)和MLP(论文中的p)'''

    def __init__(self,
                 gnn,
                 projection_hidden_size,
                 projection_size):
        '''
        Args:

        gnn: 事先定义好的gnn层
        projection_hidden_size: 通过MLP时，MLP的隐藏层维度
        projection_size: 输出维度
        '''
        super().__init__()

        self.gnn = gnn
        ## projector对应论文中的p，输入为512维，因为本文定义的嵌入表示的维度为512
        self.projector = MLP(512, projection_size, projection_hidden_size)

    def forward(self, adj, in_feats, sparse):
        representations = self.gnn(in_feats, adj, sparse)  # 初始的嵌入表示
        representations = representations.view(-1, representations.size(-1))
        projections = self.projector(representations)  # (batch, proj_dim)
        return projections


class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    '''参数更新方式，MOCO-like'''
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def sim(h1, h2):
    '''计算相似度'''
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())

####################################################################################


def agg_contrastive_loss(h, z):
    def f(x): return torch.exp(x)
    cross_sim = f(sim(h, z))
    return -torch.log(cross_sim.diag()/cross_sim.sum(dim=-1))


def interact_contrastive_loss(h1, h2):
    def f(x): return torch.exp(x)
    intra_sim = f(sim(h1, h1))
    inter_sim = f(sim(h1, h2))
    return -torch.log(inter_sim.diag() /
                      (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))

def multihead_contrastive_loss(heads, adj, tau: float = 1.0):
    loss = torch.tensor(0, dtype=float, requires_grad=True)
    for i in range(1, len(heads)):
        loss = loss + contrastive_loss(heads[0], heads[i], adj, tau=tau)
    return loss / (len(heads) - 1)
def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, adj,
                     mean: bool = True, tau: float = 1.0, hidden_norm: bool = True):
    l1 = nei_con_loss(z1, z2, tau, adj, hidden_norm)
    l2 = nei_con_loss(z2, z1, tau, adj, hidden_norm)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret


def multihead_contrastive_loss(heads, adj, tau: float = 1.0):
    loss = torch.tensor(0, dtype=float, requires_grad=True)
    for i in range(1, len(heads)):
        loss = loss + contrastive_loss(heads[0], heads[i], adj, tau=tau)
    return loss / (len(heads) - 1)


def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def nei_con_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):
    '''neighbor contrastive loss'''
    adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
    adj[adj > 0] = 1
    nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.tensor(nei_count))

    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z1, z1, hidden_norm))
    inter_view_sim = f(sim(z1, z2, hidden_norm))

    loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
            intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
    loss = loss / nei_count  # divided by the number of positive pairs for each node

    return -torch.log(loss)

class MVIEW(nn.Module):

    def __init__(self,
                 gnn,
                 feat_size,
                 projection_size,
                 projection_hidden_size,
                 prediction_size,
                 prediction_hidden_size,
                 moving_average_decay,
                 beta,
                 alpha):
        '''
        moving_average_decay: 权重更新的值
        beta: loss函数的组合
        alpha: 信息传递
        
        '''
        super().__init__()

        ## 三个分支的网络encoder初始化
        self.online_encoder = GraphEncoder(
            gnn, projection_hidden_size, projection_size)
        self.target_encoder1 = copy.deepcopy(self.online_encoder)
        self.target_encoder2 = copy.deepcopy(self.online_encoder)

        ## 目标网络不需要直接进行权重更新，接受online_encoder的权重进行MOCO-like更新
        set_requires_grad(self.target_encoder1, False)
        set_requires_grad(self.target_encoder2, False)

        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(
            projection_size, prediction_size, prediction_hidden_size)
        self.beta = beta
        self.alpha = alpha

    def reset_moving_average(self):
        del self.target_encoder1
        del self.target_encoder2
        self.target_encoder1 = None
        self.target_encoder2 = None

    def update_ma(self):
        assert self.target_encoder1 or self.target_encoder2 is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater,
                              self.target_encoder1, self.online_encoder)
        update_moving_average(self.target_ema_updater,
                              self.target_encoder2, self.online_encoder)

    def forward(self, adj, aug_adj_1, aug_adj_2, feat, aug_feat_1, aug_feat_2, sparse):
        online_proj = self.online_encoder(adj, feat, sparse)
        online_proj_1 = self.online_encoder(aug_adj_1, aug_feat_1, sparse)
        online_proj_2 = self.online_encoder(aug_adj_2, aug_feat_2, sparse)

        online_pred = self.online_predictor(online_proj)
        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)

        with torch.no_grad():
            target_proj_01 = self.target_encoder1(adj, feat, sparse)
            target_proj_11 = self.target_encoder1(aug_adj_1, aug_feat_1, sparse)
            target_proj_02 = self.target_encoder2(adj, feat, sparse)
            target_proj_22 = self.target_encoder2(
                aug_adj_2, aug_feat_2, sparse)

        l_cn_1 = self.alpha * agg_contrastive_loss(online_pred_1, target_proj_01.detach()) +\
            (1.0-self.alpha) * \
            agg_contrastive_loss(online_pred_1, target_proj_22.detach())

        l_cn_2 = self.alpha * agg_contrastive_loss(online_pred_2, target_proj_11.detach()) +\
            (1.0-self.alpha) * \
            agg_contrastive_loss(online_pred_2, target_proj_02.detach())

        l_cn = 0.5*(l_cn_1+l_cn_2)

        l_cv_0 = interact_contrastive_loss(online_pred, online_pred_1)

        l_cv_1 = interact_contrastive_loss(online_pred_1, online_pred_2)

        l_cv_2 = interact_contrastive_loss(online_pred_2, online_pred)

        l_cv = (l_cv_0+l_cv_1+l_cv_2)/3

        loss = l_cv * self.beta+l_cn * (1-self.beta)

        return loss.mean()

##################################################################################