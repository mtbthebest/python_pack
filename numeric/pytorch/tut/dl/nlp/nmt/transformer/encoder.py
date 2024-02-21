
from torch import nn
import torch

from utils import LayerNormalize, Dropout, LayerDropModuleList
from layers import TransformerEncoderLayer
from embedding import TransformerEncoderEmbedding


class TransformerEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.embed = TransformerEncoderEmbedding(cfg)

        self.layers = LayerDropModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.encoder.num_layers)])
        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNormalize(self.embed_dim)
        else:
            self.layer_norm = None
            
        self.padding_idx = cfg.padding_idx

    def forward(self, src_tokens):
        encoder_padding_mask = src_tokens.eq(self.padding_idx) # 0 for non pad 1 for pad
        
        
        x = self.embed(src_tokens)
        
        x = x * ( 1. - encoder_padding_mask.unsqueeze(-1).type_as(x))

        
        for layer in self.layers:
            x, _ = layer(x, encoder_padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x, encoder_padding_mask
