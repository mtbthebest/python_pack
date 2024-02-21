import torch
from torch import nn
import math


class FP16Optimizer:
    """
    Mixed precision optimizer with dynamic loss scaling and backoff.
    https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#scalefactor
    """
    @staticmethod
    def set_grads(params, params_with_grad):
        """
        Copies gradients from param_with_grad to params

        :param params: dst parameters
        :param params_with_grad: src parameters
        """
        for param, param_w_grad in zip(params, params_with_grad):
            if param.grad is None:
                param.grad = torch.nn.Parameter(torch.empty_like(param))
            param.grad.data.copy_(param_w_grad.grad.data)

    @staticmethod
    def set_weights(params, new_params):
        """
        Copies parameters from new_params to params

        :param params: dst parameters
        :param new_params: src parameters
        """
        for param, new_param in zip(params, new_params):
            param.data.copy_(new_param.data)

    def __init__(self, model, grad_clip=float('inf'), loss_scale=8192,
                 dls_downscale=2, dls_upscale=2, dls_upscale_interval=128, max_scale=8192):
        """
        Constructor for the Fp16Optimizer.

        :param model: model
        :param grad_clip: coefficient for gradient clipping, max L2 norm of the
            gradients
        :param loss_scale: initial loss scale
        :param dls_downscale: loss downscale factor, loss scale is divided by
            this factor when NaN/INF occurs in the gradients
        :param dls_upscale: loss upscale factor, loss scale is multiplied by
            this factor if previous dls_upscale_interval batches finished
            successfully
        :param dls_upscale_interval: interval for loss scale upscaling
        """
        print('Initializing fp16 optimizer')
        self.initialize_model(model)

        self.since_last_invalid = 0
        self.loss_scale = loss_scale
        self.dls_downscale = dls_downscale
        self.dls_upscale = dls_upscale
        self.dls_upscale_interval = dls_upscale_interval
        self.grad_clip = grad_clip
        self.max_scale = max_scale

    def initialize_model(self, model):
        """
        Initializes internal state and build fp32 master copy of weights.

        :param model: fp16 model
        """
        self.fp32_params = [param.clone().detach()
                            for param in model.parameters()]
        print('Converting model to half precision')
        model.half()
        print('Initializing fp32 clone weights')
        self.model = model
        self.model.zero_grad()
        for param in self.fp32_params:
            param.requires_grad = True

    def step(self, loss, float_loss, optimizer, update, scheduler=None, grad_scale=1):
        """
        Performs one step of the optimizer.
        Applies loss scaling, computes gradients in fp16, converts gradients to
        fp32, inverts scaling and applies optional gradient norm clipping.
        If gradients are finite, it applies update to fp32 master weights and
        copies updated parameters to fp16 model for the next iteration. If
        gradients are not finite, it skips the batch and adjusts scaling factor
        for the next iteration.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
        loss *= self.loss_scale
        loss.backward()
        is_successful = True

        if update:
            self.set_grads(self.fp32_params, self.model.parameters())
            if self.loss_scale != 1.0:
                for param in self.fp32_params:
                    param.grad.data /= (self.loss_scale * grad_scale)

            norm = nn.utils.clip_grad_norm_(self.fp32_params, self.grad_clip)

            if math.isfinite(norm):
                optimizer.step()
                self.set_weights(self.model.parameters(),
                                 self.fp32_params)
                self.since_last_invalid += 1
                if scheduler is not None:
                    scheduler.step()
            else:
                self.loss_scale /= self.dls_downscale
                self.loss_scale = max(self.loss_scale, 1)
                self.since_last_invalid = 0
                print(f'Gradient norm: {norm}')
                print(f'Skipped batch, new scale: {self.loss_scale}')
                print("Loss value: ", loss.float())
                # print("Float Loss value: ", float_loss.float())
                is_successful = False

            if self.since_last_invalid >= self.dls_upscale_interval:
                self.loss_scale *= self.dls_upscale
                self.loss_scale = min(self.loss_scale, self.max_scale)
                print(f'Upscaling, new scale: {self.loss_scale}')
                self.since_last_invalid = 0

            self.model.zero_grad()
        
        return is_successful
