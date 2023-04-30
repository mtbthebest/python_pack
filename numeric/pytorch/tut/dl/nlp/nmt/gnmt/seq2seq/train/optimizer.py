
import torch
from torch import nn


class FP16Optimizer:
    MAX_SCALE = 8192

    def __init__(self, model, grad_clip=float('inf'), loss_scale=8192,
                 downscale=2, upscale=2, upscale_interval=128) -> None:
        self.initialize_model(model)

        self.since_last_invalid = 0
        self.downscale = downscale
        self.upscale = upscale
        self.upscale_interval = upscale_interval
        self.grad_clip = grad_clip
        self.loss_scale = loss_scale

    def initialize_model(self, model):
        print("Model to half precision")
        model.half()
        self.model = model
        self.model.zero_grad()
        self.fp32_params = [param.to(torch.float32).detach() for param in self.model.parameters()]

        for param in self.fp32_params:
            param.requires_grad = True

    def set_grads(self, params, params_with_grad):
        for param, param_w_grad in zip(params, params_with_grad):
            if param.grad is None:
                param.grad = nn.Parameter(torch.empty_like(param))
            param.grad.data.copy_(param_w_grad.grad.data)

    def set_weights(self, params, new_params):
        for param, new_param in zip(params, new_params):
            param.data.copy_(new_param.data)

    def step(self, loss, optimizer, scheduler, update=True):
        loss *= self.loss_scale
        loss.backward()

        if update:
            self.set_grads(self.fp32_params, self.model.parameters())
            if self.loss_scale != 1.0:
                for param in self.fp32_params:
                    param.grad.data /= self.loss_scale
            norm = nn.utils.clip_grad_norm_(self.fp32_params, self.grad_clip)

            if torch.isfinite(norm):
                scheduler.step()
                optimizer.step()
                self.set_weights(self.model.parameters(), self.fp32_params)
                self.since_last_invalid += 1
            else:
                self.loss_scale /= self.downscale
                self.since_last_invalid = 0
                print(f"Gradient norm {norm}")
                print(f"new scale is {self.loss_scale}")

            if self.since_last_invalid >= self.upscale_interval:
                self.loss_scale *= self.upscale
                self.loss_scale = min(self.loss_scale, self.MAX_SCALE)
                print("Upscaling the loss to ", self.loss_scale)

            self.model.zero_grad()


class FP32Optimizer:

    def __init__(self, model, grad_clip=float('inf')) -> None:
        self.model = model
        self.grad_clip = grad_clip
        # self.last_epoch = 0

    def initialize_model(self, model):
        self.model = model
        self.model.zero_grad()

    def step(self, loss, optimizer, scheduler, update=True):
        loss.backward()

        if update:
            if self.grad_clip != float('inf'):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            scheduler.step()

            # scheduler.step(self.last_epoch)
            # self.last_epoch += 1

            self.model.zero_grad()
