
from typing import Optional, Dict
import torch
from torch import nn
import torch.nn.functional as F
from utils import make_positions



class LearnedPositionalEmbedding(nn.Embedding):
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx, )
        self.reset_parameters()
    
    def forward(self, input, incremental_state=None):
        if incremental_state is not None:
            positions = input.new(1, 1).fill_(self.padding_idx + input.size(1))
        else:
            positions = make_positions(input, self.padding_idx)
        return F.embedding(positions, self.weight, padding_idx=self.padding_idx)


class SinusoidalPositionalEmbedding(nn.Module):
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None):
        super().__init__()
        if padding_idx is None:
            padding_idx = 0
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.register_buffer('positions_buffer', torch.arange(padding_idx + 1, num_embeddings + padding_idx + 1))
        self.register_buffer('_float_tensor', torch.FloatTensor(1))
        self.weights = self.reset_parameters(num_embeddings, embedding_dim, padding_idx)
    
    @staticmethod
    def reset_parameters(num_embeddings, embedding_dim, padding_idx) -> torch.Tensor:
        # sin(pos / 10000 ** 2 * i / dim)
        emb_dim = embedding_dim // 2
        emb_pos = torch.log(torch.tensor(10000.)) / (emb_dim - 1)
        emb_pos = torch.exp(torch.arange(emb_dim, dtype=torch.float) * -emb_pos)
        embeddings = torch.arange(num_embeddings+1, dtype=torch.float).unsqueeze(1) * emb_pos
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)],dim=1)
        embeddings[padding_idx] = 0.0
        return embeddings
        
        
    def forward(self, input, incremental_state: Optional[Dict[str, torch.Tensor]]=None):
        # In positional embedding 0
        bz, sz = input.size()
        max_pos = self.padding_idx + 1 + sz
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.reset_parameters(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        
        self.weights = self.weights.type_as(self._float_tensor)
        if incremental_state is not None:
            positions = input.new(1, 1).fill_(self.padding_idx + input.size(1)).long().expand(bz, 1)
            sz = 1
        else:
            positions = make_positions(input, self.padding_idx)
            # positions = self.positions_buffer[:sz].expand(bz, -1) * mask_positions.type_as(input) 
        embeddings = self.weights.index_select(dim=0, index=positions.view(-1, )).view(bz, sz, self.embedding_dim)
        return embeddings.detach()
    
    
def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx):
    return SinusoidalPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)