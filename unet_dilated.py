'''
	dilated U-Net with Residual connections
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# TODO: test dilation, padding should be corresponding to dilation

class DecoderBlock(nn.Module):
    """
        outputs building block for UNet up part
    """

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    """
        outputs building block for UNet up part
    """

    def __init__(self, in_channels, middle_channels, out_channels=None, dilation=1):
        super().__init__()
        if out_channels and out_channels > 0:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, 3,
                          padding=dilation, dilation=dilation),
                nn.BatchNorm2d(middle_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_channels, out_channels, kernel_size=3,
                          stride=1, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, 3,
                          padding=dilation, dilation=dilation),
                nn.BatchNorm2d(middle_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class UNetDilated(nn.Module):
    def __init__(self, input_channels=1, class_num=1, output_func='none'):
        """
            pass
        """
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(input_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256, 256)
        self.conv4 = ConvBlock(256, 512, 512, dilation=2)
        self.conv5 = ConvBlock(512, 512, 512, dilation=2)

        self.center = DecoderBlock(
            512, 512, 256)
        self.dec5 = DecoderBlock(
            768, 512, 256)
        self.dec4 = DecoderBlock(
            768, 512, 128)
        self.dec3 = DecoderBlock(
            384, 256, 64)
        self.dec2 = DecoderBlock(
            192, 128, 32)
        self.dec1 = ConvBlock(96, 32)
        self.final = nn.Conv2d(32, class_num, kernel_size=1)
        if hasattr(nn, output_func):
            self.final = nn.Sequential(self.final, getattr(nn, output_func)())
            

    def forward(self, x):
        conv1 = self.conv1(x)  # 256
        conv2 = self.conv2(self.pool(conv1))  # 128
        conv3 = self.conv3(self.pool(conv2))  # 64
        conv4 = self.conv4(self.pool(conv3))  # 32
        conv5 = self.conv5(self.pool(conv4))  # 16

        # conv5 = self.encoder(x) # TODO: re-implement Sequential for this
        # https://discuss.pytorch.org/t/accessing-intermediate-data-in-nn-sequential/637
        # https://discuss.pytorch.org/t/extract-feature-maps-from-intermediate-layers-without-modifying-forward/1390

        center = self.center(self.pool(conv5))
        # print(center.shape, conv5.shape)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)
