
import torch
from torch import nn
import torch.nn.functional as F


def make_positions(input, padding_idx):
    mask = input.ne(padding_idx)
    positions = mask.int().cumsum(dim=1)
    positions = positions * mask.long()
    return positions


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    assert padding_idx == 0
    emb_net = nn.Embedding(num_embeddings,
                           embedding_dim,
                           padding_idx=padding_idx)
    nn.init.normal_(emb_net.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(emb_net.weight[padding_idx], 0.)
    return emb_net


class Dropout(nn.Module):

    def __init__(self, p=0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p > 0. and self.training:
            return F.dropout(x, self.p, training=self.training)
        else:
            return x


def LayerNormalize(embed_dim):
    return nn.LayerNorm(embed_dim)


class LayerDropModuleList(nn.ModuleList):

    def __init__(self, modules, p=0.0):
        super().__init__(modules=modules)
        self.p = p

    def __iter__(self):
        for module in super().__iter__():
            if not self.training or torch.rand(1).item() > self.p:
                yield module


def Linear(in_features, out_features, bias=True):
    lin_net = nn.Linear(in_features, out_features, bias=bias)
    nn.init.xavier_uniform_(lin_net.weight, )
    if lin_net.bias is not None:
        nn.init.constant_(lin_net.bias, 0.0)
    return lin_net
