import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .modules import *


class ResNet(nn.Module):
    def __init__(self, block, layers, dilate=False):
        super(ResNet, self).__init__()

        self.in_planes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=dilate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=dilate)

        # to be replaced by custom head
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, in_channels, layer_n, stride=1, dilate=False):
        downsample = None
        prev_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != in_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, in_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, in_channels, downsample, stride, dilate=prev_dilation))

        self.in_planes = in_channels * block.expansion
        for _ in range(1, layer_n):
            layers.append(block(self.in_planes, in_channels, dilate=prev_dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.reshape(x.shape[0], -1)
        # x = self.fc(x)

        return x


class ResNet18(nn.Module):
    def __init__(self, header, **kwargs):
        super(ResNet18, self).__init__()
        self.model = ResNet(ResNetBlock, [2, 2, 2, 2])
        self.header = header(**kwargs)

    def forward(self, x):
        x = self.model(x)
        x = self.header(x)

        return x


class ResNet50(nn.Module):
    def __init__(self, header, **kwargs):
        super(ResNet50, self).__init__()
        self.header = header(**kwargs)
        dilate = isinstance(self.header, DensityMapHeader)
        self.model = ResNet(ResNetBottleNeck, [3, 4, 6, 3])

    def forward(self, x):
        x = self.model(x)
        x = self.header(x)

        return x


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


class YOLOv5S(nn.Module):
    def __init__(self, header, **kwargs):
        super(YOLOv5S, self).__init__()
        self.conv_1 = Conv(3, 32, 6, 2, 2)
        self.conv_2 = Conv(32, 64, 3, 2)
        self.c3_1 = C3(64, 64, 1)
        self.conv_3 = Conv(64, 128, 3, 2)
        self.c3_2 = C3(128, 128, 2)  #
        self.conv_4 = Conv(128, 256, 3, 2)
        self.c3_3 = C3(256, 256, 3)  #
        self.conv_5 = Conv(256, 512, 3, 2)
        self.c3_4 = C3(512, 512, 1)
        self.sppf = SPPF(512, 512, 5)

        self.conv_6 = Conv(512, 256, 1, 1)  #
        self.up_1 = nn.Upsample(None, 2, 'nearest')
        self.concat_1 = Concat(1)
        self.c3_5 = C3(512, 256, 1, False)

        self.conv_7 = Conv(256, 128, 1, 1)  #
        self.up_2 = nn.Upsample(None, 2, 'nearest')
        self.concat_2 = Concat(1)
        self.c3_6 = C3(256, 128, 1, False)

        self.conv_8 = Conv(128, 128, 3, 2)
        self.concat_3 = Concat(1)
        self.c3_7 = C3(256, 256, 1, False)

        self.conv_9 = Conv(256, 256, 3, 2)
        self.concat_4 = Concat(1)
        self.c3_8 = C3(512, 512, 1, False)

        self.header = header(**kwargs)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.c3_1(x)
        x = self.conv_3(x)
        x = self.c3_2(x)
        x_4 = x.clone()
        x = self.conv_4(x)
        x = self.c3_3(x)
        x_6 = x.clone()
        x = self.conv_5(x)
        x = self.c3_4(x)
        x = self.sppf(x)

        x = self.conv_6(x)
        x_10 = x.clone()
        x = self.up_1(x)
        x = self.concat_1([x, x_6])
        x = self.c3_5(x)

        x = self.conv_7(x)
        x_14 = x.clone()
        x = self.up_2(x)
        x = self.concat_2([x, x_4])
        x = self.c3_6(x)

        x = self.conv_8(x)
        x = self.concat_3([x, x_14])
        x = self.c3_7(x)

        x = self.conv_9(x)
        x = self.concat_4([x, x_10])
        x = self.c3_8(x)

        x = self.header(x)

        return x


class UNet(nn.Module):
    def __init__(self, header, **kwargs):
        super(UNet, self).__init__()
        self.bilinear = False

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.header = header(**kwargs)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.header(x)
        return x
