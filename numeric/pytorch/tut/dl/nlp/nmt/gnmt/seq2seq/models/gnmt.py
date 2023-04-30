
from torch import nn
from seq2seq.data import config
from seq2seq.models.seq2seq_base import Seq2Seq
from seq2seq.models.encoder import ResidualRecurrentEncoder
from seq2seq.models.decoder import ResidualRecurrentDecoder


class GNMT(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=1024,
                 num_layers=4, dropout=0.2, batch_first=False, share_embedding=True) -> None:
        super().__init__(batch_first=batch_first)

        if share_embedding:
            embedder = nn.Embedding(vocab_size, hidden_size,
                                    padding_idx=config.PAD)
        else:
            embedder = None

        self.encoder = ResidualRecurrentEncoder(vocab_size, hidden_size, num_layers, dropout,
                                                batch_first, embedder)

        self.decoder = ResidualRecurrentDecoder(vocab_size, hidden_size, num_layers, dropout,
                                                batch_first, embedder)

    def forward(self, input_encoder, input_enc_len, input_decoder):
        context = self.encode(input_encoder, input_enc_len)
        context = (context, input_enc_len, None)
        output, _, _ = self.decode(input_decoder, context)
        return output
