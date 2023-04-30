import itertools

import torch
from torch import nn
from seq2seq.data import config
from seq2seq.models.attention import BahdanauAttention


def init_lstm(lstm, init_weight=0.1):
    nn.init.uniform_(lstm.weight_hh_l0.data, -init_weight, init_weight)
    nn.init.uniform_(lstm.weight_ih_l0.data, -init_weight, init_weight)

    nn.init.uniform_(lstm.bias_ih_l0.data, -init_weight, init_weight)
    nn.init.zeros_(lstm.bias_hh_l0.data)

    if lstm.bidirectional:
        nn.init.uniform_(lstm.weight_hh_l0_reverse.data, -init_weight, init_weight)
        nn.init.uniform_(lstm.weight_ih_l0_reverse.data, -init_weight, init_weight)

        nn.init.uniform_(lstm.bias_ih_l0_reverse.data, -init_weight, init_weight)
        nn.init.zeros_(lstm.bias_hh_l0_reverse.data)


class RecurrentAttention(nn.Module):

    def __init__(self, input_size=1024, context_size=1024, hidden_size=1024, num_layers=1, batch_first=False,
                 dropout=0.2, init_weight=0.1) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=batch_first)
        init_lstm(self.rnn, init_weight)
        self.attn = BahdanauAttention(hidden_size, context_size, context_size,
                                      normalize=True, batch_first=batch_first)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, hidden, context, context_len):
        self.attn.set_mask(context_len, context)

        inputs = self.dropout(inputs)
        rnn_outputs, hidden = self.rnn(inputs, hidden)
        attn_outputs, scores = self.attn(rnn_outputs, context)

        return rnn_outputs, hidden, attn_outputs, scores


class Classifier(nn.Module):

    def __init__(self, in_features, out_features, init_weight=0.1) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_features, out_features)
        nn.init.uniform_(self.classifier.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.classifier.bias.data, -init_weight, init_weight)

    def forward(self, x):
        out = self.classifier(x)
        return out


class ResidualRecurrentDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers=4, dropout=0.2, batch_first=False,
                 embedder=None, init_weight=0.1) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.att_rnn = RecurrentAttention(hidden_size, hidden_size, hidden_size, num_layers=1,
                                          batch_first=batch_first, dropout=dropout)
        self.rnn_layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.rnn_layers.append(nn.LSTM(2*hidden_size, hidden_size, num_layers=1,
                                   bias=True, batch_first=batch_first))

        for lstm in self.rnn_layers:
            init_lstm(lstm, init_weight)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab_size, hidden_size, padding_idx=config.PAD)
            nn.init.uniform_(self.embedder.weight.data, -init_weight, init_weight)
        self.classifier = Classifier(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, hidden):
        if hidden is not None:
            hidden = hidden.chunk(self.num_layers)
            hidden = tuple(i.chunk(2) for i in hidden)
        else:
            hidden = [None] * self.num_layers

        self.next_hidden = []

        return hidden

    def append_hidden(self, h):
        if self.inference:
            self.next_hidden.append(h)

    def package_hidden(self):
        if self.inference:
            hidden = torch.cat(tuple(itertools.chain(*self.next_hidden)))
        else:
            hidden = None
        return hidden

    def forward(self, inputs, context, inference=False):
        self.inference = inference

        enc_context, enc_len, hidden = context

        hidden = self.init_hidden(hidden)

        x = self.embedder(inputs)
        x, h, attn, scores = self.att_rnn(x, hidden[0], enc_context, enc_len)

        self.append_hidden(h)

        x = torch.cat((x, attn), dim=2)
        x = self.dropout(x)
        x, h = self.rnn_layers[0](x, hidden[1])
        self.append_hidden(h)

        for i in range(1, len(self.rnn_layers)):
            residual = x
            x = torch.cat((x, attn), dim=2)
            x = self.dropout(x)
            x, h = self.rnn_layers[i](x, hidden[i+1])
            self.append_hidden(h)
            x = x + residual

        x = self.classifier(x)
        hidden = self.package_hidden()
        return x, scores, [enc_context, enc_len, hidden]
