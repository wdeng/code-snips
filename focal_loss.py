# https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    r"""
        This criterion is a implementation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
        
            Loss(x, class) = - \alpha (1-sigmoid(x)[class])^gamma \log(sigmoid(x)[class])
        The loss function is implemented as a weighted sigmoid BCE loss for 2D and 3D images
    
        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classified examples (p > .5), 
                                   putting more focus on hard, mis-classified examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_weights=1, gamma=1.5, per_class=False, size_average=True):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.per_class = per_class
        self.size_average = size_average

    def forward(self, inputs, targets):
        p = inputs.sigmoid()  # or softmax if use NLLLoss
        pt = p * targets + (1 - p) * (1 - targets) # pt = p if t > 0 else 1-p
        weights = self.class_weights * (1 - pt).pow(self.gamma)
        if self.per_class:
            shape = list(inputs.shape)
            N, C = shape[:2]
            for i in range(2, len(shape)):
                shape[i] = 1
            weights = weights.view(N, C, -1).mean(dim=-1).view(*shape) # TODO: is weights N correct?

        return F.binary_cross_entropy_with_logits(inputs, targets, weights, size_average=self.size_average)
