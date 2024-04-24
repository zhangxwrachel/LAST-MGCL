# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
os.chdir('/root/MGCL')
import numpy as np
import scipy.sparse as sp
import torch
import random
import argparse
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from utils import process
from utils import aug
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from modules.gat import GAT
from net.mview_gat import MVIEW
import time
import torch.nn.functional as F
import dgl
from augf.Citation.cvae_models import VAE

import sys
import copy
import torch.optim as optim
# import augf.Citation.cvae_pretrain
from augf.Citation.utils import accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
from tqdm import trange
# from NCLA.gat import GAT
from NCLA.utils import load_network_data, get_train_data, random_planetoid_splits
from NCLA.loss import multihead_contrastive_loss

import dgl
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--dataset', type=str, default='citeseer')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--eval_every', type=int, default=10)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)#0.01
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--sample_size', type=int, default=2000)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--sparse', type=str_to_bool, default=True)
parser.add_argument('--input_dim', type=int, default=3327)
parser.add_argument('--proj_dim', type=int, default=512)
parser.add_argument('--proj_hid', type=int, default=4096)
parser.add_argument('--pred_dim', type=int, default=512)
parser.add_argument("--pretrain_lr", type=float, default=0.01)
parser.add_argument('--pred_hid', type=int, default=4096)
parser.add_argument('--momentum', type=float, default=0.8)
parser.add_argument('--beta', type=float, default=0.3)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument("--gpu", type=int, default=0,help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--num-heads", type=int, default=2,help="number of hidden attention heads")
parser.add_argument("--num-layers", type=int, default=1,help="number of hidden layers")
parser.add_argument("--num-hidden", type=int, default=128,help="number of hidden units")                    #
parser.add_argument("--tau", type=float, default=1,help="temperature-scales")
parser.add_argument("--in-drop", type=float, default=0.6,help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.5,help="attention dropout")
parser.add_argument('--weight-decay', type=float, default=1e-4,help="weight decay")
parser.add_argument('--negative-slope', type=float, default=0.2,help="the negative slope of leaky relu")
args = parser.parse_known_args()[0]
torch.set_num_threads(4)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()
adj, features, Y = load_network_data(args.dataset)
features[features > 0] = 1
g = dgl.from_scipy(adj)

if args.gpu >= 0 and torch.cuda.is_available():
    cuda = True
    g = g.int().to(args.gpu)
else:
    cuda = False

features = torch.FloatTensor(features.todense())##################################################################################################

f = open('NCLA_' + args.dataset + '.txt', 'a+')
f.write('\n\n\n{}\n'.format(args))
f.flush()

labels = np.argmax(Y, 1)
print('labels:',labels.shape)
adj = torch.tensor(adj.todense())

all_time = time.time()
num_feats = features.shape[1]
n_classes = Y.shape[1]
n_edges = g.number_of_edges()

# add self loop
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)

# create model
heads = ([args.num_heads] * args.num_layers)
model = GAT(g,
            args.num_layers,
            num_feats,
            args.num_hidden,
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.negative_slope)

mview = MVIEW(gnn=model,
                feat_size=args.input_dim,
                projection_size=args.proj_dim,
                projection_hidden_size=args.proj_hid,
                prediction_size=args.pred_dim,
                prediction_hidden_size=args.pred_hid,
                moving_average_decay=args.momentum, beta=args.beta, alpha=args.alpha).to(device)
if cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
# use optimizer
# optimizer = torch.optim.Adam(
#     model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = torch.optim.Adam(
    [
        {'params': list(model.parameters()) + list(mview.parameters()), 'lr': args.lr, 'weight_decay': args.weight_decay}
    ]
)
# initialize graph
dur = []
test_acc = 0
counter = 0
min_train_loss = 100
early_stop_counter = 100
best_t = -1
#开始进行训练
import torch
import random
ori_features = features
svd_u,s,svd_v = torch.svd(features)      #可以进行参数分析
u_mul_s = svd_u @ (torch.diag(s))
v_mul_s = svd_v @ (torch.diag(s))
# del s
print('SVD done.')
k1 =6# 保留的奇异值数量#############################################################################
# 保留较大的奇异值和相应的特征向量
U_truncated = svd_u[:, :k1]
S_truncated = torch.diag(s[:k1])
VT_truncated = svd_v.t()[:k1,:]
# 重建矩阵
aug_features1 = torch.matmul(U_truncated, torch.matmul(S_truncated, VT_truncated))
new_path = 'augf/Citation'
# 改变当前路径
os.chdir(new_path)
import cvae_pretrain
exc_path = sys.path[0]
cvae_model = torch.load("model/{}.pkl".format(args.dataset))
cvae_model.to(device)
cvae_features = features
z = torch.randn([cvae_features.size(0), cvae_model.latent_size]).to(device)
augmented_features = cvae_model.inference(z, cvae_features)
aug_features2 = cvae_pretrain.feature_tensor_normalize(augmented_features).detach()
######改回路径
os.chdir('/root/MGCL')
ori_features = features
for epoch in range(args.epochs):
    if epoch >= 0:
        t0 = time.time()
    model.train()
    mview.train()
    optimizer.zero_grad()
    # heads = model(features)
    _,loss = mview(adj, ori_features, aug_features1, aug_features2)
    loss.backward(retain_graph=True)
    optimizer.step()
    mview.update_ma()
    ori_features = features
######上面是模型训练，下面是模型评估
    model.eval()
    # mview.eval()
    with torch.no_grad():
        online_pred,loss_train = mview(adj, ori_features, aug_features1, aug_features2)
        # loss_train = multihead_contrastive_loss(heads, adj, tau=args.tau)
    # early stop if loss does not decrease for 100 consecutive epochs
    if loss_train < min_train_loss:
        counter = 0
        min_train_loss = loss_train
        best_t = epoch
        torch.save(model.state_dict(), 'best_GAT.pkl')
        torch.save(model.state_dict(), 'best_NCLA.pkl')
    else:
        counter += 1
    if counter >= early_stop_counter:
        print('early stop')
        break
    if epoch >= 0:
        dur.append(time.time() - t0)
    print("Epoch {:04d} | Time(s) {:.4f} | TrainLoss {:.4f} ".
          format(epoch + 1, np.mean(dur), loss_train.item()))
print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_NCLA.pkl'))
##############################
model.eval()#将模型设置为评估模式，并通过模型传递特征以获得嵌入
with torch.no_grad():
    heads = model(features)
    ################
embeds = torch.cat(heads, axis=1)  #从所有头部中串联嵌入
embeds = embeds.detach().cpu()
Accuaracy_test_allK = []
numRandom = 20
for train_num in [20]:
    AccuaracyAll = []
    for random_state in range(numRandom):
        print(
            "\n=============================%d-th random split with training num %d============================="##################################
            % (random_state + 1, train_num))
        if train_num == 20:
            if args.dataset in ['cora', 'citeseer', 'pubmed']:
                # train_num per class: 20, val_num: 500, test: 1000
                val_num = 500
                idx_train, idx_val, idx_test = random_planetoid_splits(n_classes, torch.tensor(labels), train_num,
                                                                       random_state)
            else:
                # Coauthor CS, Amazon Computers, Amazon Photo
                # train_num per class: 20, val_num per class: 30, test: rest
                val_num = 30
                idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

        else:
            val_num = 0  # do not use a validation set when the training labels are extremely limited
            idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

        train_embs = embeds[idx_train, :]
        val_embs = embeds[idx_val, :]
        test_embs = embeds[idx_test, :]

        if train_num == 20:
            # find the best parameter C using validation set
            best_val_score = 0.0
            for param in [1e-4, 1e-3, 1e-2, 0.1, 1,10,100]:
                LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, C=param)
                LR.fit(train_embs, labels[idx_train])
                val_score = LR.score(val_embs, labels[idx_val])
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_parameters = {'C': param}
            LR_best = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, **best_parameters)
            LR_best.fit(train_embs, labels[idx_train])
            y_pred_test = LR_best.predict(test_embs)  # pred label
            print("Best accuracy on validation set:{:.4f}".format(best_val_score))#################
            print("Best parameters:{}".format(best_parameters))#########
        else:  # not use a validation set when the training labels are extremely limited
            LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)
            LR.fit(train_embs, labels[idx_train])
            y_pred_test = LR.predict(test_embs)  # pred label
        test_acc = accuracy_score(labels[idx_test], y_pred_test)
        print("test accuaray:{:.4f}".format(test_acc))###############################################
        AccuaracyAll.append(test_acc)
    average_acc = np.mean(AccuaracyAll) * 100
    std_acc = np.std(AccuaracyAll) * 100
    print('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
        numRandom, average_acc, std_acc, train_num, val_num))
    f.write('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
        numRandom, average_acc, std_acc, train_num, val_num))
    f.flush()
    Accuaracy_test_allK.append(average_acc)
f.close()