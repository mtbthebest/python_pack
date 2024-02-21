
from collections import defaultdict
from functools import partial
import torch

import subword_nmt.apply_bpe
import sacremoses
from config import Config as config



class Tokenizer:
    
    def __init__(self, vocab_fname=None, bpe_fname=None, lang=None, pad=1, separator="@@") -> None:
        self.separator = separator
        self.lang = lang

        if bpe_fname:
            with open(bpe_fname, encoding="utf-8") as bpe:
                self.bpe = subword_nmt.apply_bpe.BPE(bpe)
        if vocab_fname:   
            self.build_vocabulary(vocab_fname, pad)
        if lang:
            self.init_moses(lang)
    
    def init_moses(self, lang):
        self.moses_tokenizer = sacremoses.MosesTokenizer(lang['src'])
        self.moses_detokenizer = sacremoses.MosesDetokenizer(lang['tgt'])

    def build_vocabulary(self, vocab_fname, pad):
        print(f"Building vocabulary {vocab_fname}")
        vocab = [config.PAD_TOKEN, config.UNK_TOKEN, config.BOS_TOKEN, config.EOS_TOKEN]
        
        with open(vocab_fname) as vfile:
            for line in vfile:
                vocab.append(line.strip().split()[0])
        
        self.pad_vocabulary(vocab, pad)
        
        self.vocab_size = len(vocab)
        
        print(f"Vocabulary size {self.vocab_size}")
        
        self.tok2idx = defaultdict(partial(int, config.UNK_TOKEN))
        
        for idx, token in enumerate(vocab):
            self.tok2idx[token] = idx
        
        self.idx2tok = {}
        
        for token, idx in self.tok2idx.items():
            self.idx2tok[idx] = token
    
    def pad_vocabulary(self, vocab, pad):
        vocab_size = len(vocab)
        padded_vocab_size = (vocab_size + pad - 1) // pad * pad
        
        for i in range(0, padded_vocab_size - vocab_size):
            token = f"madeupword{i:04d}"
            vocab.append(token)
        
        assert len(vocab) % pad == 0
    
    def get_state(self):
        return {
            "lang": self.lang,
            "separator": self.separator,
            "vocab_size": self.vocab_size,
            "bpe": self.bpe,
            "tok2idx": self.tok2idx,
            "idx2tok": self.idx2tok
        }
    
    def set_state(self, state):
        self.lang = state['state']
        self.separator = state['separator']
        self.vocab_size = state['vocab_size']
        self.bpe = state['bpe']
        self.tok2idx = state['tok2idx']
        self.idx2tok = state['idx2tok']
        
        self.init_moses(self.lang)
    
    def segment(self, line):
        line = line.strip().split()
        entry = [self.tok2idx[token] for token in line]
        entry = [config.BOS] + entry + [config.EOS]
        return entry

    def tokenize(self, line):
        tokenized = self.moses_tokenizer.tokenize(line, return_str=True)
        bpe = self.bpe.process_line(tokenized)
        segmented = self.segment(bpe)
        tensor = torch.tensor(segmented)
        
        return tensor
        
    def detokenize_moses(self, line):
        return self.moses_detokenizer.detokenize(line.split())
    
    def detokenize_bpe(self, inp, delim=' '):
        detok = delim.join([self.idx2tok[idx] for idx in inp])
        detok = detok.replace(self.separator + " ", "")
        detok = detok.replace(self.separator, '')
        
        for key in [config.BOS_TOKEN, config.EOS_TOKEN, config.PAD_TOKEN]:
            detok = detok.replace(key, "")
        
        detok = detok.strip()
        
        return detok
        
    
    def detokenize(self, inp):
        detok_inp = self.detokenize_bpe(inp)
        return self.detokenize_moses(detok_inp)
    