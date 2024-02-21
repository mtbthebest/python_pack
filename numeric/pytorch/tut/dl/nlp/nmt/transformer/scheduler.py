
import torch
import math
from torch.optim.optimizer import Optimizer


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer: Optimizer,
                 warmup_steps,
                 total_steps,
                 warmup_lr=5e-5,
                 decay_factor=0.4) -> None:

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.decay_factor = decay_factor
        self.decay_div = 2 * (self.total_steps - warmup_steps)
        self.decay_epoch = 0
        super().__init__(optimizer, verbose=False)

    def get_lr(self):
        # print("LAst epoch ", self.last_epoch)
        if self.last_epoch <= self.warmup_steps:
            return [self.get_lrs_warmup_stage(
                lr, self.warmup_lr, self.last_epoch, self.warmup_steps)
                for lr in self.base_lrs]
        else:
            self.decay_epoch = self.last_epoch - self.warmup_steps
            return [lr * (1 - (self.decay_epoch / self.decay_div) ** self.decay_factor) for lr in self.base_lrs]

    def get_lrs_warmup_stage(self, lr, warmup_lr, last_epoch, warmup_steps):
        constant = 5
        ratio = math.log(lr / (lr - warmup_lr))
        a = warmup_steps / (constant - ratio)
        b = a * ratio
        return lr * (1 - math.exp(-(last_epoch + b) / a))
