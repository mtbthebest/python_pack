
import json
import codecs
import torch
import torchaudio
from torchaudio.compliance import kaldi
from fairseq.data import FairseqDataset


def normalize(feat: torch.Tensor):
    var, mean = torch.var_mean(feat, dim=0)
    var.clip_(min=torch.tensor(1e-8))
    return (feat - mean).div(var)


class ASRDataset(FairseqDataset):

    def __init__(self, fname, asr_dict, num_mel_bins=80,
                 win_size=25.0, stride=10.0,):
        super().__init__()
        samples = sorted(json.load(codecs.open(fname)).items(),
                         key=lambda sample: sample[1]["input"]["length_ms"],
                         reverse=True)
        self.audio_fnames = [s[1]["input"]["path"] for s in samples]
        self.audio_ids = [s[0] for s in samples]
        self.frame_sizes = [s[1]["input"]["length_ms"] for s in samples]
        self.labels = [[int(i) for i in s[1]["output"]["tokenid"].split(", ")]
                       for s in samples]
        self.labels = [[*t, asr_dict.eos()] for t in self.labels]
        self.num_mel_bins = num_mel_bins
        self.win_size = win_size
        self.stride = stride

    def __getitem__(self, index):
        label = self.labels[index]
        wf, sr = torchaudio.load(self.audio_fnames[index])
        feat = kaldi.fbank(wf, frame_length=self.win_size, sample_frequency=sr,
                           frame_shift=self.stride, num_mel_bins=self.num_mel_bins,
                           window_type='hamming')
        feat = normalize(feat)
        return {"id": index,  'audio': feat, 'text': label}

    def __len__(self):
        return len(self.audio_fnames)
