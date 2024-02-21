
import torch
import math
from torch.optim.optimizer import _use_grad_for_differentiable


class LAMB(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, max_norm=10., betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, differentiable=False) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, max_norm=max_norm,
                        eps=eps, differentiable=differentiable)
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        super().__init__(params, defaults=defaults)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]['steps'])
        if not step_is_tensor:
            for s in state_values:
                s['steps'] = torch.tensor(float(s['steps']))

    @_use_grad_for_differentiable
    def step(self, closure=None):
        # TODO
        # return 
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.update(group)

        return loss

    def update(self, group):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']
        max_norm = group['max_norm']
        for param in group['params']:
            if param.grad is not None:
                grad = param.grad

                if param not in self.state:
                    exp_avg = torch.zeros_like(
                        param, dtype=param.dtype, device=param.device, memory_format=torch.preserve_format)
                    exp_avg_norm = torch.zeros_like(
                        param, dtype=param.dtype, device=param.device, memory_format=torch.preserve_format)

                    self.state[param]['exp_avg'] = exp_avg
                    self.state[param]['exp_avg_norm'] = exp_avg_norm
                    self.state[param]['steps'] = torch.tensor(0.)

                exp_avg = self.state[param]['exp_avg']
                exp_avg_norm = self.state[param]['exp_avg_norm']
                steps = self.state[param]['steps']

                steps += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_norm.mul_(beta2).add_(grad * grad, alpha=1-beta2)

                bias_correction1 = 1 - beta1 ** steps
                bias_correction2 = 1 - beta2 ** steps

                if not torch.is_tensor(bias_correction2):
                    bias_correction2_sqrt = math.sqrt(bias_correction2)
                else:
                    bias_correction2_sqrt = bias_correction2.sqrt()

                denom = (exp_avg_norm.sqrt() / bias_correction2_sqrt) + eps
                numer = exp_avg / bias_correction1

                update = numer / denom

                if weight_decay != 0:
                    update = update.add(param, alpha=weight_decay)

                trust_ratio = 1.0
                param_t_norm = param.norm()
                grad_t_norm = update.norm()
                if param_t_norm > 0 and grad_t_norm > 0:
                    trust_ratio = (
                        param_t_norm / grad_t_norm).clip(max=max_norm)

                step_size = trust_ratio * lr

                param.add_(-step_size * update)
