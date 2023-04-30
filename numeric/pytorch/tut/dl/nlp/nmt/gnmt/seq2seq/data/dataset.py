
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from seq2seq.data import config
from seq2seq.data.sampler import RandomSampler, BucketSampler


def build_collate_fn(batch_first=False, parallel=True, sort=False):

    def collate_seq(seq):
        lengths = torch.tensor([len(s) for s in seq]).to(torch.int64)
        batch_length = max(lengths)

        shape = (len(seq), batch_length)

        seq_tensor = torch.full(shape, config.PAD, dtype=torch.int64)

        for i, s in enumerate(seq):
            end_seq = lengths[i]
            seq_tensor[i, :end_seq].copy_(s[:end_seq])

        if not batch_first:
            seq_tensor = seq_tensor.t()

        return seq_tensor, lengths

    def parallel_collate(seqs):
        src_seqs, tgt_seqs = zip(*seqs)
        if sort:
            indices, src_seqs = zip(*sorted(enumerate(src_seqs), key=lambda item: len(item[1]), reverse=True))
            tgt_seqs = [tgt_seqs[idx] for idx in indices]

        return tuple([collate_seq(s) for s in [src_seqs, tgt_seqs]])

    def single_collate(src_seqs):
        if sort:
            indices, src_seqs = zip(*sorted(enumerate(src_seqs), key=lambda item: len(item[1]), reverse=True))
        else:
            indices = range(len(src_seqs))

        return collate_seq(src_seqs), tuple(indices)

    if parallel:
        return parallel_collate
    else:
        return single_collate


class TextDataset(Dataset):

    def __init__(self, src_fname, tokenizer, min_len=None, max_len=None, sort=False, max_size=None) -> None:
        self.min_len = min_len
        self.max_len = max_len
        self.parallel = False
        self.sorted = False
        self.tokenizer = tokenizer

        self.src = self.process_data(src_fname, max_size)

        if min_len is not None and max_len is not None:
            self.filter_data(min_len, max_len)

        lengths = [len(s) for s in self.src]
        self.length = torch.tensor(lengths)

        if sort:
            self.sort_by_length()

    def process_data(self, fname, max_size):
        print(f"Processing file from {fname}")
        data = []
        with open(fname) as sfile:
            for idx, line in enumerate(sfile):
                if max_size and idx == max_size:
                    break
                entry = self.tokenizer.segment(line)

                entry = torch.tensor(entry)
                data.append(entry)

        return data

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index]

    def filter_data(self, min_len, max_len):
        print(f"Filtering between {min_len} {max_len}")
        initial_len = len(self.src)

        filtered_src = []

        for src in self.src:
            if min_len <= len(src) <= max_len:
                filtered_src.append(src)

        self.src = filtered_src
        filtered_len = len(self.src)

        print(f"Pairs before: {initial_len}, after: {filtered_len}")

    def sort_by_length(self):
        self.lengths, indices = self.length.sort(descending=True)
        self.src = [self.src[idx] for idx in indices]

        self.indices = indices.tolist()
        self.sorted = True

    def unsort(self, array):
        if self.sorted:
            inverse = sorted(enumerate(self.indices), key=lambda x: x[1])
            array = [array[i[0]] for i in inverse if i]
        return array

    def get_loader(self, batch_size=1, seeds=None, shuffle=False, num_workers=0, batch_first=False,
                   pad=False, batching=None, batching_opt={}, drop_last=False):
        collate_fn = build_collate_fn(batch_first, parallel=self.parallel, sort=True)
        if shuffle:
            if batching == "bucketing":
                sampler = BucketSampler(self, batch_size, seeds, batching_opt['num_buckets'])
            elif batching == "random":
                sampler = RandomSampler(self, batch_size, seeds)
        else:
            sampler = None

        return DataLoader(self,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          sampler=sampler,
                          num_workers=num_workers,
                          #   pin_memory=True,
                          drop_last=drop_last)


class LazyParallelDataset(TextDataset):

    def __init__(self, src_fname, tgt_fname, tokenizer, min_len, max_len, sort=False, max_size=None) -> None:
        self.min_len = min_len
        self.max_len = max_len
        self.parallel = True
        self.sorted = False
        self.tokenizer = tokenizer

        self.raw_src = self.process_raw_data(src_fname, max_size)
        self.raw_tgt = self.process_raw_data(tgt_fname, max_size)

        self.filter_raw_data(min_len - 2, max_len - 2)
        assert len(self.raw_src) == len(self.raw_tgt)

        self.src_lengths = torch.tensor([i + 2 for i in self.src_len])
        self.tgt_lengths = torch.tensor([i + 2 for i in self.tgt_len])

        self.lengths = self.src_lengths + self.tgt_lengths

    def process_raw_data(self, fname, max_size):
        data = []
        with open(fname) as dfile:
            for idx, line in enumerate(dfile):
                if max_size and idx == max_size:
                    break
                data.append(line)
        return data

    def filter_raw_data(self, min_len, max_len):
        initial_len = len(self.raw_src)
        filtered_src, filtered_tgt, filtered_src_len, filtered_tgt_len = [[] for _ in range(4)]

        for src, tgt in zip(self.raw_src, self.raw_tgt):
            src_len = src.count(" ") + 1
            tgt_len = tgt.count(" ") + 1

            if min_len <= src_len <= max_len and min_len <= tgt_len <= max_len:
                filtered_src.append(src)
                filtered_tgt.append(tgt)
                filtered_src_len.append(src_len)
                filtered_tgt_len.append(tgt_len)

        self.raw_src = filtered_src
        self.raw_tgt = filtered_tgt
        self.src_len = filtered_src_len
        self.tgt_len = filtered_tgt_len

    def __getitem__(self, index):
        src = torch.tensor(self.tokenizer.segment(self.raw_src[index]))
        tgt = torch.tensor(self.tokenizer.segment(self.raw_tgt[index]))
        return src, tgt

    def __len__(self):
        return len(self.raw_src)


class ParallelDataset(TextDataset):
    def __init__(self, src_fname, tgt_fname, tokenizer, min_len, max_len, sort=False, max_size=None) -> None:
        self.min_len = min_len
        self.max_len = max_len
        self.parallel = True
        self.sorted = False
        self.tokenizer = tokenizer

        self.src = self.process_data(src_fname, max_size)
        self.tgt = self.process_data(tgt_fname, max_size)

        assert len(self.src) == len(self.tgt)

        self.filter_data(min_len, max_len)
        assert len(self.src) == len(self.tgt)

        self.src_lengths = torch.tensor([len(s) for s in self.src])
        self.tgt_lengths = torch.tensor([len(s) for s in self.tgt])

        self.lengths = self.src_lengths + self.tgt_lengths

        if sort:
            self.sort_by_length()

    def sort_by_length(self):
        self.lengths, indices = self.lengths.sort(descending=True)
        self.src = [self.src[idx] for idx in indices]
        self.tgt = [self.tgt[idx] for idx in indices]

        self.src_lengths = [self.src_lengths[idx] for idx in indices]
        self.tgt_lengths = [self.tgt_lengths[idx] for idx in indices]

        self.indices = indices.tolist()
        self.sorted = True

    def filter_data(self, min_len, max_len):
        filtered_src, filtered_tgt = [[] for _ in range(2)]

        for src, tgt in zip(self.src, self.tgt):
            src_len = len(src)
            tgt_len = len(tgt)

            if min_len <= src_len <= max_len and min_len <= tgt_len <= max_len:
                filtered_src.append(src)
                filtered_tgt.append(tgt)

        self.src = filtered_src
        self.tgt = filtered_tgt

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]
