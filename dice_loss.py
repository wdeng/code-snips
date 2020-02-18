# https://github.com/pytorch/pytorch/issues/1249

import torch.nn.functional as F
import torch

from torch.nn import Module, BCELoss

EPS = 1E-6
SMOOTH = 1.


def soft_dice(predict, target):

    i_flat = predict.view(-1)
    t_flat = target.view(-1)
    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + SMOOTH) /
                (i_flat.sum() + t_flat.sum() + SMOOTH))


class DiceLoss(Module):
    def forward(self, predict, target):
        return soft_dice(predict, target)


class DiceBCELoss(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bce = BCELoss()

    def forward(self, predict, target):
        return soft_dice(predict, target) + self.bce(predict, target)


class DiceLossChannelWeight(Module):
    def forward(self, predict, target, weights):
        intersection, union = 0., 0.
        for i, w in enumerate(weights):
            i_flat = predict[:, i, ...].view(-1)
            t_flat = target[:, i, ...].view(-1)
            intersection += (i_flat * t_flat).sum() * w
            union += (i_flat.sum() + t_flat.sum()) * w

        return 1. - ((2. * intersection + SMOOTH) / (union + SMOOTH))
