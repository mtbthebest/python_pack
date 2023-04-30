
import math
import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer,
                 iterations,
                 warmup_steps=0,
                 remain_steps=1,
                 decay_interval=None,
                 decay_steps=4,
                 decay_factor=0.5,
                 last_epoch=-1) -> None:
        print("Starting warmup scheduler ", warmup_steps)

        self.warmup_steps = warmup_steps * iterations
        self.remain_steps = iterations * remain_steps
        if decay_interval is None:
            decay_iterations = iterations - self.remain_steps
            self.decay_interval = decay_iterations // decay_steps
            self.decay_interval = max(self.decay_interval, 1)
        else:
            self.decay_interval = decay_interval * iterations
        print(self.warmup_steps, self.decay_interval, self.remain_steps, iterations)
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor

        if self.warmup_steps > self.remain_steps:
            self.warmup_steps = self.remain_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # print("last epoch ", self.last_epoch, self.warmup_steps,  self.remain_steps)
        
        # TODO
        # if self.last_epoch <= self.warmup_steps:
        #     if self.warmup_steps != 0:
        #         warmup_factor = math.exp(math.log(0.01) / self.warmup_steps)
        #         # print("Selmf warmup ", warmup_factor, self.warmup_steps, self.last_epoch)
        #     else:
        #         warmup_factor = 1.0

        #     inv_decay = warmup_factor ** (self.warmup_steps - self.last_epoch)
        #     lr = [base_lr * inv_decay for base_lr in self.base_lrs]
        # elif self.last_epoch >= self.remain_steps:
        #     decay_iter = self.last_epoch - self.remain_steps
        #     num_decay_steps = decay_iter // self.decay_interval + 1
        #     num_decay_steps = min(num_decay_steps, self.decay_steps)
        #     lr = [
        #         base_lr * (self.decay_factor ** num_decay_steps)
        #         for base_lr in self.base_lrs
        #     ]
        # else:
        #     lr = [base_lr for base_lr in self.base_lrs]
        # print("Lr values is: ", lr[0])
        # return lr
        #TODO
        return [9e-4 for base_lr in self.base_lrs]