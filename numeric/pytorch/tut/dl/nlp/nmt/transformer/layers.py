
from torch import nn
from multi_head_attention import MultiHeadAttention
from utils import Dropout, LayerNormalize, Linear


class TransformerEncoderLayer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.layer_norm1 = LayerNormalize(cfg.encoder.embed_dim)

        self.multi_attn = MultiHeadAttention(cfg.encoder.embed_dim,
                                             cfg.encoder.num_heads,
                                             self_attn=True,
                                             dropout=cfg.encoder.attention_dropout)

        self.dropout = Dropout(cfg.encoder.dropout)
        self.activation_dropout = Dropout(cfg.encoder.activation_dropout)
        self.activation_fn = nn.ReLU()

        self.fn1 = Linear(cfg.encoder.embed_dim,
                          cfg.encoder.ffn_embed_dim, bias=True)
        self.fn2 = Linear(cfg.encoder.ffn_embed_dim,
                          cfg.encoder.embed_dim, bias=True)

        self.layer_norm2 = LayerNormalize(cfg.encoder.embed_dim)

        self.normalize_before = cfg.encoder.normalize_before

    def forward(self, x, key_padding_mask):
        residual = x
        if self.normalize_before:
            x = self.layer_norm1(x)

        x, attn_probs = self.multi_attn(
            query=x, key=x, value=x, key_padding_mask=key_padding_mask)
        x = self.dropout(x)
        x = residual + x
        if not self.normalize_before:
            x = self.layer_norm1(x)

        residual = x

        if self.normalize_before:
            x = self.layer_norm2(x)

        x = self.activation_fn(self.fn1(x))
        x = self.activation_dropout(x)
        x = self.fn2(x)

        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm2(x)

        return x, attn_probs


class TransformerDecoderLayer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.decoder.embed_dim
        self.ffn_embed_dim = cfg.decoder.ffn_embed_dim
        self.self_attn = MultiHeadAttention(self.embed_dim,
                                            cfg.decoder.num_heads,
                                            self_attn=True,
                                            dropout=cfg.decoder.attention_dropout)
        self.enc_dec_attn = MultiHeadAttention(self.embed_dim,
                                               cfg.decoder.num_heads,
                                               kdim=cfg.encoder.embed_dim,
                                               vdim=cfg.encoder.embed_dim,
                                               enc_dec_attn=True,
                                               dropout=cfg.decoder.attention_dropout)

        self.dropout = Dropout(cfg.decoder.dropout)
        self.activation_dropout = Dropout(cfg.decoder.activation_dropout)
        self.activation_fn = nn.ReLU()
        self.fn1 = Linear(self.embed_dim, self.ffn_embed_dim, bias=True)
        self.fn2 = Linear(self.ffn_embed_dim, self.embed_dim, bias=True)

        self.layer_norm1 = LayerNormalize(self.embed_dim)
        self.layer_norm2 = LayerNormalize(self.embed_dim)
        self.layer_norm_fc = LayerNormalize(self.embed_dim)

        self.normalize_before = cfg.decoder.normalize_before

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state=None,
                key_padding_mask=None, attn_masks=None):
        residual = x
        if self.normalize_before:
            x = self.layer_norm1(x)

        x, dec_attn_probs = self.self_attn(query=x, key=x, value=x,
                                           key_padding_mask=key_padding_mask,
                                           attn_masks=attn_masks,
                                           incremental_state=incremental_state)
        x = self.dropout(x)
        x = residual + x
        if not self.normalize_before:
            x = self.layer_norm1(x)

        residual = x

        if self.normalize_before:
            x = self.layer_norm2(x)

        x, _ = self.enc_dec_attn(query=x,
                                 key=encoder_out,
                                 value=encoder_out,
                                 key_padding_mask=encoder_padding_mask,
                                 incremental_state=incremental_state,
                                 static_kv=True)
        x = self.dropout(x)
        x = x + residual
        if not self.normalize_before:
            x = self.layer_norm2(x)

        residual = x

        if self.normalize_before:
            x = self.layer_norm_fc(x)
        x = self.activation_fn(self.fn1(x))
        x = self.activation_dropout(x)
        x = self.fn2(x)
        x = self.dropout(x)

        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm_fc(x)

        return x, dec_attn_probs
