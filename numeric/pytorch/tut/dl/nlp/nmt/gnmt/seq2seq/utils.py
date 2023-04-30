
import random

def generate_seed(num_seeds, seed=0):
    seeding_rnd = random.Random(seed)
    
    shuffling_seeds = [seeding_rnd.randint(0, 2**32 - 1) for _ in range(num_seeds)]
    
    return shuffling_seeds

class AverageMeter:
    def __init__(self, warmup=0, keep=False) -> None:
        self.reset()
        self.warmup = warmup
        self.keep = keep
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.iters = 0
        self.vals = []
    
    def update(self, val, n=1):
        self.iters += 1
        self.val = val
        
        if self.iters > self.warmup:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            if self.keep:
                self.vals.append(val)
        