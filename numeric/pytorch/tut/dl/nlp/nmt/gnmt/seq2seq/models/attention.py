

import math

import torch
from torch import nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):

    def __init__(self, query_size, key_size, num_units, normalize=True, batch_first=False, init_weight=0.1) -> None:
        super().__init__()
        self.normalize = normalize
        self.batch_first = batch_first
        self.num_units = num_units

        self.linear_q = nn.Linear(query_size, num_units, bias=False)
        self.linear_k = nn.Linear(key_size, num_units, bias=False)
        nn.init.uniform_(self.linear_q.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.linear_k.weight.data, -init_weight, init_weight)

        self.linear_att = nn.Parameter(torch.Tensor(num_units))

        self.mask = None

        if self.normalize:
            self.normalize_scalar = nn.Parameter(torch.Tensor(1))
            self.normalize_bias = nn.Parameter(torch.Tensor(num_units))
        else:
            self.register_parameter('normalize_scalar', None)
            self.register_parameter('normalize_bias', None)

    def reset_parameters(self, init_weight):
        stdv = 1.0 / math.sqrt(self.num_units)
        self.linear_att.data.uniform_(-init_weight, init_weight)

        if self.normalize:
            self.normalize_scalar.data.fill_(stdv)
            self.normalize_bias.data.zero_()

    def set_mask(self, context_len, context):
        if self.batch_first:
            max_len = context.size(1)
        else:
            max_len = context.size(0)

        indices = torch.arange(0, max_len, dtype=torch.int64, device=context.device)
        self.mask = indices >= (context_len.unsqueeze(1))

    def calc_score(self, att_query, att_keys):
        batch_size, key_size, feat_dim = att_keys.size()
        query_size = att_query.size(1)

        att_query = att_query.unsqueeze(2).expand(batch_size, query_size, key_size, feat_dim)
        att_keys = att_keys.unsqueeze(1).expand(batch_size, query_size, key_size, feat_dim)

        sum_query_keys = att_query + att_keys
        if self.normalize:
            sum_query_keys = sum_query_keys + self.normalize_bias
            linear_att = self.linear_att / self.linear_att.norm()
            linear_att = linear_att * self.normalize_scalar
        else:
            linear_att = self.linear_att

        out = torch.tanh(sum_query_keys).matmul(linear_att)
        return out

    def forward(self, query, keys):
        if not self.batch_first:
            keys = keys.transpose(0, 1)
            if query.dim() == 3:
                query = query.swapaxes(0, 1)

        if query.dim == 2:
            single_query = True
            query = query.unsqueeze(1)
        else:
            single_query = False

        batch_size, key_size, query_size = query.size(0), keys.size(1), query.size(1)

        processed_query = self.linear_q(query)
        processed_key = self.linear_k(keys)

        scores = self.calc_score(processed_query, processed_key)  # (bs, qs, ks)

        if self.mask is not None:  # (bs, ks)
            mask = self.mask.unsqueeze(1).expand(batch_size, query_size, key_size)
            scores.masked_fill_(mask, -65504.)
        # (bs, qs, ks)
        scores_normalized = F.softmax(scores, dim=-1)
        # (bs, qs, ks) * (bs, ks, fs) => (bs, qs, fs)
        context = torch.bmm(scores, keys)

        if single_query:
            context = context.squeeze(1)
            scores_normalized = scores_normalized.squeeze(1)
        elif not self.batch_first:
            context = context.swapaxes(0, 1)
            scores_normalized = scores_normalized.transpose(0, 1)

        return context, scores_normalized
