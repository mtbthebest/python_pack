import os
import numpy as np
import struct
from PIL import Image
from torch.utils.data import Dataset
import torch


# means = (109.58, 97.37, 74.80)
# means = [0.42972549, 0.38184314, 0.29333333]
# means = 0.429
normalize_mean = 109.5
# std = (42.25, 34.35, 35.97)
# std = [0.16568627, 0.13470588, 0.14105882]
# std = 0.16
normalize_std  = 42.25


class Oxford102Flowers(Dataset):
    path = os.path.join("/mnt/dl", "datasets", "Oxford102FlowersSplits")
    
    def __getitem__(self, index):
        x = self.imgs[index].astype(np.float32)
        x_target = x.copy()
        label = self.labels[index]
        if self.transform is not None: 
            x_target = self.transform(x_target)
            x = np.resize(x, (x_target.size(1), x_target.size(2), x_target.size(0)))
        x_target -= normalize_mean
        x_target /= normalize_std
        x = torch.from_numpy(x).permute(2, 0, 1)
        x /= 255.0
        return x, x_target, label
    
    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def load_image(fname):
        with Image.open(fname) as img:
            img = img.convert('RGB')
        return np.array(img).astype(np.uint8)
    
    
    @staticmethod
    def load_data(path, which):
        images = []
        labels = []
        for data_type in which:
            files = sorted(map(lambda f: os.path.join(path, data_type, "jpeg", f), os.listdir(os.path.join(path, data_type, "jpeg"))))
            with open(os.path.join(path, data_type, "label", "label.txt")) as f:
                for i, label_value in enumerate(f.readlines()):
                    labels.append(int(label_value))
                    images.append(Oxford102Flowers.load_image(files[i]))   
        return images, np.array(labels).astype(np.int32)
    

class Oxford102FlowersTrain(Oxford102Flowers):
    
    def __init__(self, transform=None, which=["train", "test"]) -> None:
        self.transform = transform
        self.imgs, self.labels = self.load_data(self.path, which)
        

class Oxford102FlowersTest(Oxford102Flowers):
    
    def __init__(self, transform=None) -> None:
        self.transform = transform
        self.imgs, self.labels = self.load_data(self.path, ["valid"])