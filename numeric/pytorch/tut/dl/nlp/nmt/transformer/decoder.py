
import torch
from torch import nn
import math
from typing import Optional, Dict

from positional_encoding import PositionalEmbedding
from utils import LayerNormalize, Dropout, LayerDropModuleList, Linear
from layers import TransformerDecoderLayer
from embedding import TransformerDecoderEmbedding


class TransformerDecoder(nn.Module):
    def __init__(self, cfg, bias=False) -> None:
        super().__init__()
        self.padding_idx = cfg.padding_idx

        self.embed = TransformerDecoderEmbedding(cfg)

        self.layers = LayerDropModuleList(
            [TransformerDecoderLayer(cfg) for _ in range(cfg.decoder.num_layers)])

        if cfg.decoder.normalize_before:
            self.final_layernorm = LayerNormalize(cfg.decoder.embed_dim)
        else:
            self.final_layernorm = None

        self.fc = Linear(cfg.decoder.embed_dim,
                         cfg.decoder.vocab_size, bias=bias)

    def get_future_masks(self, tokens, tokens_embs):
        masks_tensor = torch.tril(tokens.new(
            tokens.size(1), tokens.size(1)).fill_(0.), 0).float()
        masks = torch.arange(tokens.size(1)).unsqueeze(1).lt(
            torch.arange(tokens.size(1))).to(tokens.device).to(bool)
        # masks_tensor = masks_tensor.masked_fill(masks, float('-inf'))
        masks_tensor = masks_tensor.masked_fill(masks, -65536.)
        return masks_tensor.type_as(tokens_embs)

    def forward(self, tgt_tokens, encoder_out, encoder_padding_mask : Optional[torch.Tensor]=None,
                incremental_state: Optional[Dict[str, torch.Tensor]]=None):

        key_padding_mask = tgt_tokens.eq(self.padding_idx)

        x = self.embed(tgt_tokens, incremental_state=incremental_state)

        if incremental_state is not None:
            key_padding_mask = key_padding_mask[:, -1:]
            attn_masks = None
        else:
            attn_masks = self.get_future_masks(tgt_tokens, x)

        for layer in self.layers:
            x, _ = layer(x,
                         encoder_out,
                         encoder_padding_mask,
                         incremental_state=incremental_state,
                         key_padding_mask=key_padding_mask,
                         attn_masks=attn_masks)

        if self.final_layernorm:
            x = self.final_layernorm(x)

        x = self.fc(x)
        return x
