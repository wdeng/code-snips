import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# from libs.net_models import upsample_add
# from libs.net_models.vaes import reparametrize


# https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
# https://github.com/jwyang/fpn.pytorch/blob/master/lib/model/fpn/fpn.py

def upsample_add(x, y, mode='bilinear'):
    '''Upsample and add two feature maps.
    Args:
        x: (Variable) top feature map to be upsampled.
        y: (Variable) lateral feature map.
    Returns:
        (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    '''
    align_corners = None if mode == 'nearest' else False
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode=mode, align_corners=align_corners) + y


class ResFPN50(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, output_func='none', pretrained=False, use_vars=False):
        super().__init__()

        encoder = models.resnext50_32x4d(pretrained)

        self.pool = encoder.maxpool

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        # if pretrained:
        #     for param in self.parameters():
        #         param.requires_grad = False

        if input_channels == 3:
            self.conv1 = nn.Sequential(
                encoder.conv1, encoder.bn1, encoder.relu)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                       nn.BatchNorm2d(64),
                                       encoder.relu)

        # Reduce channels, maybe not 64, but 128 or 256
        self.center = nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0)
        self.dec4 = nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0)
        self.dec3 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.dec2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)

        # self.final = nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        # if use_vars: output_channels *= 2
        # self.reparam = use_vars

        self.final = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
                                   nn.BatchNorm2d(32),
                                   encoder.relu,
                                   nn.Conv2d(32, output_channels, kernel_size=3, padding=1)
                                   )
        if hasattr(nn, output_func):
            self.final = nn.Sequential(self.final, getattr(nn, output_func)())

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.layer1(self.pool(conv1))
        conv3 = self.layer2(conv2)
        conv4 = self.layer3(conv3)
        conv5 = self.layer4(conv4)

        center = self.center(conv5)
        dec4 = upsample_add(center, self.dec4(conv4))
        dec3 = upsample_add(dec4, self.dec3(conv3))
        dec2 = upsample_add(dec3, self.dec2(conv2))
        dec1 = upsample_add(dec2, conv1)

        out = self.final(dec1)

        # if self.reparam:
        #     z_dim = out.shape[1] // 2
        #     out = reparametrize(out[:, :z_dim, ...],out[:, z_dim:, ...]) if self.training else out[:, :z_dim, ...]

        return out

