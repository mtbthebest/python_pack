
import uuid
import torch
from torch import nn
import torch.nn.functional as F
import math

from utils import Linear


class QueryLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, weights):
        # print("Forward: ")
        # print("inputs: ", inputs.dtype)
        q = inputs.view(-1, inputs.size(2),) @ weights
        ctx.save_for_backward(inputs, weights)
        return q.view(inputs.size(0), inputs.size(1), -1).detach()

    @staticmethod
    def backward(ctx, grad_q):
        inputs, weights = ctx.saved_tensors
        grad_input, grad_weight = None, None
        if ctx.needs_input_grad[0]:
            try:
                grad_input = grad_q.type_as(weights) @  weights.t()
                # grad_input = grad_q @  weights.t()
            except Exception as e:
                # print("grad q: ", grad_q.dtype)
                # print("weights: ", weights.dtype)
                # print("inputs: ", inputs.dtype)
                raise e
        if ctx.needs_input_grad[1]:
            grad_weight = grad_q.unsqueeze(-1).type_as(inputs) * \
                inputs.unsqueeze(inputs.dim() - 1)
            grad_weight = grad_weight.sum((0, 1)).T

        return grad_input, grad_weight

@torch.jit.script
def linear_proj(inputs, weights):
    # return F.linear(inputs, weights)
    return QueryLinear.apply(inputs, weights)


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, 
                 dropout=0.0, bias=False, self_attn=False, enc_dec_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self.dropout = dropout
        self.self_attn = self_attn
        self.enc_dec_attn = enc_dec_attn
        
        self.in_q_weight_proj = nn.Parameter(
            torch.Tensor(embed_dim, embed_dim))
        self.in_k_weight_proj = nn.Parameter(torch.Tensor(self.kdim, embed_dim))
        self.in_v_weight_proj = nn.Parameter(torch.Tensor(self.vdim, embed_dim))
        self.scaling = 1. / math.sqrt(self.head_dim)

        if bias:
            self.in_q_bias_proj = nn.Parameter(torch.Tensor(embed_dim))
            self.in_k_bias_proj = nn.Parameter(torch.Tensor(embed_dim))
            self.in_v_bias_proj = nn.Parameter(torch.Tensor(embed_dim))
        else:
            self.register_parameter('in_q_bias_proj', None)
            self.register_parameter('in_k_bias_proj', None)
            self.register_parameter('in_v_bias_proj', None)

        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

        self.reset_parameters()
        
        self._incremental_state_id = str(uuid.uuid4())


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_q_weight_proj)
        nn.init.xavier_uniform_(self.in_k_weight_proj)
        nn.init.xavier_uniform_(self.in_v_weight_proj)
        if self.in_k_bias_proj:
            nn.init.constant_(self.in_q_bias_proj, 0.)
            nn.init.constant_(self.in_k_bias_proj, 0.)
            nn.init.constant_(self.in_v_bias_proj, 0.)

    def get_input_buffer(self, incremental_state):
        if incremental_state is not None and self._incremental_state_id in incremental_state:
            return incremental_state[self._incremental_state_id]
        return dict()
    
    def save_input_buffer(self, incremental_state, saved_state):
        incremental_state[self._incremental_state_id] = saved_state
        return incremental_state
    
    def get_incremental_mask(self, key_padding_mask, prev_key_padding_mask, static_kv):
        if static_kv and prev_key_padding_mask is not None:
            return prev_key_padding_mask
        elif key_padding_mask is not None and prev_key_padding_mask is not None:
            return torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        elif prev_key_padding_mask is not None:
            raise NotImplementedError
            return prev_key_padding_mask
        elif key_padding_mask is not None:
            return key_padding_mask
        else:
            # TODO
            raise NotImplementedError
            return prev_key_padding_mask
    
    
    def forward(self, 
                query,
                key,
                value,
                key_padding_mask=None,
                attn_masks=None,
                incremental_state=None,
                static_kv=False):
        q = linear_proj(query, self.in_q_weight_proj)

        k = v = saved_state = None
        
        if incremental_state is not None:
            saved_state = self.get_input_buffer(incremental_state)
            if saved_state and "prev_key" in saved_state and static_kv:
                assert self.enc_dec_attn and not self.self_attn
                key = value = None

        if self.self_attn:
            k = linear_proj(query, self.in_k_weight_proj)
            v = linear_proj(query, self.in_v_weight_proj)
        elif self.enc_dec_attn:
            if key is not None:
                k = linear_proj(key, self.in_k_weight_proj)
                v = linear_proj(key, self.in_v_weight_proj)
        else:
            k = linear_proj(key, self.in_k_weight_proj)
            v = linear_proj(value, self.in_v_weight_proj)
        bz, qz, _ = query.size()
        # bhtd
        q = q.view(bz, qz, self.num_heads, self.head_dim).transpose(1, 2)
        
        if saved_state is not None:
            if "prev_key" in saved_state:
                prev_key = saved_state['prev_key'] 
                if static_kv:
                    assert k is None
                    k = prev_key
                else:
                    k = torch.cat([prev_key, k], dim=1)
            
            if "prev_value" in saved_state:
                prev_value = saved_state['prev_value'] 
                if static_kv:
                    assert v is None
                    v = prev_value
                else:
                    v = torch.cat([prev_value, v], dim=1)
    
            prev_key_padding_mask = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            key_padding_mask = self.get_incremental_mask(key_padding_mask=key_padding_mask,
                                                         prev_key_padding_mask=prev_key_padding_mask,
                                                         static_kv=static_kv)
            saved_state['prev_key'] = k
            saved_state['prev_value'] = v
            saved_state['prev_key_padding_mask'] = key_padding_mask
            
            self.save_input_buffer(incremental_state, saved_state)
        
        assert k is not None
        bz, kz, _ = k.size()
        k = k.view(bz, kz, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bz, kz, self.num_heads, self.head_dim).transpose(1, 2)
        
        # batch_size, head_num, tgt_len, src_len
        qk = torch.einsum('bhtd, bhsd -> bhts', q, k)
        qk = qk * self.scaling
        if attn_masks is not None:
            qk += attn_masks
        if key_padding_mask is not None:
            # qk.masked_fill_(key_padding_mask.expand(self.num_heads, -1, -1).transpose(0, 1).contiguous().unsqueeze(3), float('-inf'))
                qk.float().masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -65536.)
                qk = qk.type_as(q)
            
        attn_probs = F.softmax(qk, dim=-1, dtype=qk.dtype)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        attn_v = torch.einsum('bhts, bhsd -> bhtd', attn_probs, v)
        
        
        attn_v = attn_v.transpose(1, 2).contiguous().view(bz, qz, self.embed_dim)
        attn_v_proj = self.out_proj(attn_v)
        
        return attn_v_proj, attn_probs.mean(dim=1)
        
