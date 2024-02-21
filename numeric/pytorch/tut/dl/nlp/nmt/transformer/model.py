

import torch
from torch import nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder


class TransformerModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(cfg)
        self.decoder = TransformerDecoder(cfg)

    def forward(self, src_tokens, tgt_tokens):
        encoder_out, encoder_padding_mask = self.encoder(src_tokens)
        decoder_out = self.decoder(
            tgt_tokens, encoder_out, encoder_padding_mask, )
        return decoder_out
