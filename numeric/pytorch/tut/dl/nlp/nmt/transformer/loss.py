
from torch import nn
import torch.nn.functional as F


# class SoftmaxLoss(nn.Module):

#     def __init__(self, padding_idx=0, reduction='mean'):
#         super().__init__()
#         self.padding_idx = padding_idx
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         bz, sz, vz = inputs.size()
#         pad_mask = targets.view(-1, ) != self.padding_idx
#         non_pad_targets = targets.view(-1, ).masked_select(pad_mask)
#         non_pad_inputs = inputs.view(-1, vz).contiguous().masked_select(
#             pad_mask.unsqueeze(1)).view(non_pad_targets.size(0), -1)
#         return F.cross_entropy(input=non_pad_inputs, target=non_pad_targets, 
#                                reduction=self.reduction)



def SoftmaxLoss(padding_idx, reduction):

    def criterion(inputs, targets):
        bz, sz, vz = inputs.size()
        pad_mask = targets.view(-1, ) != padding_idx
        non_pad_targets = targets.view(-1, ).masked_select(pad_mask)
        non_pad_inputs = inputs.view(-1, vz).contiguous().masked_select(
            pad_mask.unsqueeze(1)).view(non_pad_targets.size(0), -1)
        return F.cross_entropy(input=non_pad_inputs, target=non_pad_targets, 
                               reduction=reduction)
    
    return criterion


class SoftmaxScalerLoss(nn.Module):
    
    def __init__(self, padding_idx=0, reduction='mean'):
        super().__init__()
        self.padding_idx = padding_idx
        self.reduction = reduction

    def forward(self, inputs, targets, scale):
        bz, sz, vz = inputs.size()
        pad_mask = targets.view(-1, ) != self.padding_idx
        non_pad_targets = targets.view(-1, ).masked_select(pad_mask)
        non_pad_inputs = inputs.view(-1, vz).contiguous().masked_select(
            pad_mask.unsqueeze(1)).view(non_pad_targets.size(0), -1)
        return F.cross_entropy(input=non_pad_inputs, 
                               target=non_pad_targets, 
                               reduction=self.reduction) / scale