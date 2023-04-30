
import torch
from torch import nn
from seq2seq.data import config

class ResidualRecurrentEncoder(nn.Module):
    
    def __init__(self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2, batch_first=False, 
                 embedder=None, init_weight=0.1) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.rnn_layers = nn.ModuleDict()
        
        self.rnn_layers['layer_1'] = nn.LSTM(input_size=hidden_size, 
                                             hidden_size=hidden_size,
                                             num_layers=1, 
                                             bias=True,
                                             batch_first=batch_first,
                                             bidirectional=True)
        self.rnn_layers['layer_2'] = nn.LSTM((2*hidden_size), hidden_size, num_layers=1,
                                             bias=True, batch_first=batch_first)
        for lay in range(num_layers - 2):
            self.rnn_layers[f'layer_{lay + 3}'] = nn.LSTM(hidden_size, hidden_size, num_layers=1,
                                             bias=True, batch_first=batch_first)
        
        for _, lstm in self.rnn_layers.items():
            self.init_lstm(lstm, init_weight)
        
        self.dropout = nn.Dropout(p=dropout)
        
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab_size, hidden_size, padding_idx=config.PAD)
            
            nn.init.uniform_(self.embedder.weight.data, -init_weight, init_weight)
            
    
    def forward(self, inputs, lengths):
        x = self.embedder(inputs)
        x = self.dropout(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().numpy(), batch_first=self.batch_first)
        x, _ = self.rnn_layers['layer_1'](x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=self.batch_first)
        
        x = self.dropout(x)
        x, _ = self.rnn_layers['layer_2'](x)
        
        for i in range(3, len(self.rnn_layers) + 1):
            residual = x
            x = self.dropout(x)
            x, _ = self.rnn_layers[f"layer_{i}"](x)
            x = x + residual
        
        return x

    def init_lstm(self, lstm, init_weight=0.1):
        nn.init.uniform_(lstm.weight_hh_l0.data, -init_weight, init_weight)
        nn.init.uniform_(lstm.weight_ih_l0.data, -init_weight, init_weight)
        
        nn.init.uniform_(lstm.bias_ih_l0.data, -init_weight, init_weight)
        nn.init.zeros_(lstm.bias_hh_l0.data)
        
        if lstm.bidirectional:
            nn.init.uniform_(lstm.weight_hh_l0_reverse.data, -init_weight, init_weight)
            nn.init.uniform_(lstm.weight_ih_l0_reverse.data, -init_weight, init_weight)
            
            nn.init.uniform_(lstm.bias_ih_l0_reverse.data, -init_weight, init_weight)
            nn.init.zeros_(lstm.bias_hh_l0_reverse.data)