import os
import numpy as np
import struct

from torch.utils.data import Dataset


class MNIST(Dataset):
    path = os.path.join("/mnt/dl", "EMNIST", "emnist_source_files")
    
    def __getitem__(self, index):
        img = self.imgs[index].astype(np.float32)
        label = self.labels[index]
        if self.transform is not None: 
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def load_mnist(path, which='train'):
    
        if which == 'train':
            labels_path = os.path.join(path, 'emnist-digits-train-labels-idx1-ubyte')
            images_path = os.path.join(path, 'emnist-digits-train-images-idx3-ubyte')
        elif which == 'test':
            labels_path = os.path.join(path, 'emnist-digits-test-labels-idx1-ubyte')
            images_path = os.path.join(path, 'emnist-digits-test-images-idx3-ubyte')
        else:
            raise AttributeError('`which` must be "train" or "test"')
            
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, n, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = images.reshape(-1, int(np.sqrt(images.shape[1])), int(np.sqrt(images.shape[1])))
        return images, labels.astype(np.int64)
    

class MNISTTrain(MNIST):
    
    def __init__(self, transform=None) -> None:
        self.transform = transform
        self.imgs, self.labels = self.load_mnist(self.path, "train")
        

class MNISTTest(MNIST):
    
    def __init__(self, transform=None) -> None:
        self.transform = transform
        self.imgs, self.labels = self.load_mnist(self.path, "test")