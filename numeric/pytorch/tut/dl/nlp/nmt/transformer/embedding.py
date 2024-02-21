
import math
from torch import nn
import torch
from typing import Dict, Optional

from utils import Embedding, LayerNormalize, Dropout, LayerDropModuleList

from positional_encoding import PositionalEmbedding


class TransformerEncoderEmbedding(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        embed_dim = cfg.encoder.embed_dim
        
        self.embed_tokens = Embedding(num_embeddings=cfg.encoder.vocab_size,
                                      embedding_dim=cfg.encoder.embed_dim,
                                      padding_idx=cfg.padding_idx)
        self.embed_pos = PositionalEmbedding(num_embeddings=cfg.seq_length,
                                             embedding_dim=cfg.encoder.embed_dim,
                                             padding_idx=cfg.padding_idx)
        self.dropout_embedding = Dropout(cfg.encoder.dropout_embedding)
        
        self.layernorm_embedding = LayerNormalize(
            embed_dim) if cfg.encoder.layernorm_embedding else None
        
        self.embed_scale = math.sqrt(
            embed_dim) if cfg.encoder.scale_embedding else 1.0

    def forward(self, tokens,):
        x = self.embed_tokens(tokens) * self.embed_scale
        x = x + self.embed_pos(tokens)
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = self.dropout_embedding(x)
        return x
    


class TransformerDecoderEmbedding(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        embed_dim = cfg.decoder.embed_dim
        
        self.embed_tokens = Embedding(num_embeddings=cfg.decoder.vocab_size,
                                      embedding_dim=cfg.decoder.embed_dim,
                                      padding_idx=cfg.padding_idx)
        self.embed_pos = PositionalEmbedding(num_embeddings=cfg.seq_length,
                                             embedding_dim=cfg.decoder.embed_dim,
                                             padding_idx=cfg.padding_idx)
        self.dropout_embedding = Dropout(cfg.decoder.dropout_embedding)
        
        self.layernorm_embedding = LayerNormalize(
            embed_dim) if cfg.decoder.layernorm_embedding else None
        
        self.embed_scale = math.sqrt(
            embed_dim) if cfg.decoder.scale_embedding else 1.0

    def forward(self, tokens, incremental_state: Optional[Dict[str, torch.Tensor]]=None):
        positions = self.embed_pos(tokens, incremental_state=incremental_state)
        if incremental_state is not None:
            tokens = tokens[:, -1:, ]
        x = self.embed_tokens(tokens) * self.embed_scale
        x = x + positions
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = self.dropout_embedding(x)
        return x
