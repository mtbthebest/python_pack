
import torch
from torch import nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    
    def __init__(self, padding_idx, smoothing=0.0) -> None:
        super().__init__()
        
        self.padding_idx = padding_idx
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
    
    def forward(self, x, target):
        non_pad_mask = (target != self.padding_idx)
        
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, 
                                    index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)[non_pad_mask]
        smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        return loss.sum()
        
        
        
        