from torch.utils.data import Sampler 
import torch

class RandomSampler(Sampler):
    
    def __init__(self, dataset, batch_size, seeds=[0]):
        self.dataset = dataset
        data_len = len(dataset)
        
        num_samples = data_len // batch_size
        num_samples *= batch_size
        
        self.num_samples = num_samples
        
        indices = list(range(len(dataset)))
        
        indices = torch.tensor(indices, dtype=torch.int64)
        self.indices = indices
        self.seeds = seeds
        self.epoch = 0  
    
    def init_rng(self):
        rng = torch.Generator()
        
        rng.manual_seed(self.seeds[self.epoch])
        
        return rng
        
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        rng = self.init_rng()
        for k, i in enumerate(torch.randperm(len(self.indices), generator=rng)):
            if k == self.num_samples:
                break
            yield self.indices[i]
        

    def __len__(self):
        return self.num_samples
    
class BucketSampler(Sampler):
    
    def __init__(self, dataset, batch_size, seeds, num_buckets):
        self.dataset = dataset
        self.data_len = len(self.dataset)
        self.seeds = seeds
        self.num_buckets = num_buckets
        
        bucket_width = (dataset.max_len + num_buckets - 1) // num_buckets
        
        bucket_ids = torch.max(dataset.src_lengths // bucket_width,
                               dataset.tgt_lengths // bucket_width)
        bucket_ids.clamp_(0, num_buckets - 1)
        all_indices = torch.arange(self.data_len)
        
        self.buckets = []
        self.batch_size = batch_size
        
        self.num_samples = 0
        
        for bid in range(num_buckets):
            indices = all_indices[bucket_ids == bid]
            self.buckets.append(indices)
            print(f"Bucket {bid} had length {len(indices)}")
            num_samples =  len(indices) // self.batch_size * self.batch_size
            self.num_samples += num_samples
        print("New samples ", self.num_samples)
            
    def init_rng(self):
        rng = torch.Generator()
        rng.manual_seed(self.seeds[self.epoch])
        return rng
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def reshuffle_batches(self, indices, rng):
        indices = indices.view(-1, self.batch_size)
        num_batches = indices.size(0)
        order = torch.randperm(num_batches, generator=rng)
        indices = indices[order, :]
        indices = indices.view(-1)
        
        return indices
        
    
    def __iter__(self):
        rng = self.init_rng()
        
        indices = []
        for bid in range(self.num_buckets):
            perm = torch.randperm(len(self.buckets[bid]), generator=rng)
            bucket_indices = self.buckets[bid][perm]
            
            length = len(bucket_indices) // self.batch_size * self.batch_size
            # assert length >= self.batch_size
            bucket_indices = bucket_indices[:length]
            indices.append(bucket_indices)
        
        indices = torch.cat(indices)
        a =  indices.clone()
        assert len(indices) % self.batch_size == 0
        assert len(indices.view(-1)) == self.num_samples
        
        indices = self.reshuffle_batches(indices, rng)
        indices = indices.numpy()#.tolist()
        return iter(indices)
    
    def __len__(self):
        return self.num_samples 
        
        
        
        
        
        
        