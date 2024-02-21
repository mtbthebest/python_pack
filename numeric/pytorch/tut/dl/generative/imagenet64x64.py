
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler


class Imagenet64x64Dataset(Dataset):
    path = "/mnt/dl/datasets/imagenet64x64"

    def __init__(self, split, sizes=None, transforms=None):
        super().__init__()
        self.img_fnames = os.listdir(os.path.join(self.path, split))
        self.transforms = transforms
        self.sizes = sizes
        self.form_batches(sizes)

    def __getitem__(self, index):
        return index
        img_path = self.path[index]
        img = Image.open(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return self.length

    def form_batches(self, sizes):
        self.length = len(self.img_fnames)
        if sizes is None:
            return

        r = int(np.ceil(len(self.img_fnames) / sizes))
        batch_sizes = [(i*sizes, (i+1) * sizes) for i in range(r)]
        batch_sizes[-1] = batch_sizes[-1][0], min(
            batch_sizes[-1][1], self.length)

        if batch_sizes[-1][-1] - batch_sizes[-1][0] < sizes:
            batch_sizes[-2] = (batch_sizes[-2][0], self.length)
            batch_sizes.pop()
        self.length = sizes
        self.batch_sizes = batch_sizes


class Imagenet64x64Sampler(Sampler):

    def __init__(self, ds):
        super().__init__()
        self.ds = ds
        self.batch_sizes = np.arange(
            0, len(self.ds.img_fnames), dtype=np.int32).tolist()
        self.count = 0

    def __iter__(self):
        start, end = self.ds.batch_sizes[self.count]
        batch_idx = iter(self.batch_sizes[start:end])
        self.count = (self.count + 1) % len(self.ds.batch_sizes)
        if self.count == 0:
            self.reset_batch_sizes()
        return batch_idx

    def reset_batch_sizes(self):
        self.batch_sizes = np.random.choice(
            len(self.batch_sizes), len(self.batch_sizes), replace=False).tolist()
        return
