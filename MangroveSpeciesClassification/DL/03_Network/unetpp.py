# -*- coding = utf-8 -*-
# @Time : 2021/11/11 14:54
# @Author : 自在清风
# @File : unetpp.py
# @software: PyCharm
import math

import torch
from torch import nn
import torch.nn.functional as f
import time


def tow_layer_conv(in_dim, out_dim):
    layer = [nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
             nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
    return nn.Sequential(*layer)


def tow_layer_conv_with_pool(in_dim, out_dim):
    layer = [nn.MaxPool2d(kernel_size=2, stride=2),
             nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
             nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
    return nn.Sequential(*layer)


def three_layer_conv_with_pool(in_dim, middle_dim, out_dim):
    layer = [nn.MaxPool2d(kernel_size=2, stride=2),
             nn.Conv2d(in_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
             nn.Conv2d(middle_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(middle_dim), nn.ReLU(inplace=True),
             nn.Conv2d(middle_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
    return nn.Sequential(*layer)


def three_layer_conv(in_dim, middle_dim, out_dim):
    layer = [
        nn.Conv2d(in_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        nn.Conv2d(middle_dim, middle_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.BatchNorm2d(middle_dim), nn.ReLU(inplace=True),
        nn.Conv2d(middle_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
    return nn.Sequential(*layer)


def convTranspose(in_dim, out_dim):
    layer = [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=(2, 2), stride=(2, 2)),
             nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
             ]
    return nn.Sequential(*layer)


def conv3_3(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True))


def conv1_1(in_dim, out_dim):
    return nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))


def tow_conv_without_RELU(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim))


def three_conv_without_RELU(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                         nn.BatchNorm2d(out_dim))


class SELayer(nn.Module):
    def __init__(self, out_dim, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, out_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim // reduction, out_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEPreBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SEPreBlock, self).__init__()
        self.conv_without_relu = tow_conv_without_RELU(in_dim, out_dim)
        self.SE = SELayer(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_without_relu(x)
        a = x
        y = self.SE(x)
        out = self.relu(a + y)
        return out


class SEBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SEBlock, self).__init__()
        self.conv_without_relu = three_conv_without_RELU(in_dim, out_dim)
        self.SE = SELayer(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_without_relu(x)
        a = x
        y = self.SE(x)
        out = self.relu(a + y)
        return out


class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECAPreBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ECAPreBlock, self).__init__()
        self.ECA = ECALayer(in_dim)
        self.conv_without_relu = tow_conv_without_RELU(in_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_without_relu(x)
        a = x
        x = self.ECA(x)
        out = self.relu(a + x)
        return out


class ECABlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ECABlock, self).__init__()
        self.ECA = ECALayer(in_dim)
        self.conv_without_relu = three_conv_without_RELU(in_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_without_relu(x)
        a = x
        x = self.ECA(x)
        out = self.relu(a + x)
        return out


class ECAUnet(nn.Module):
    def __init__(self, color_dim=3, num_classes=2):
        super(ECAUnet, self).__init__()
        layer = [16, 32, 64, 128, 256]
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_d0 = ECAPreBlock(color_dim, layer[0])
        self.conv_d1 = ECAPreBlock(layer[0], layer[1])
        self.conv_d2 = ECABlock(layer[1], layer[2])
        self.conv_d3 = ECABlock(layer[2], layer[3])

        self.conv_d4 = ECABlock(layer[3], layer[4])

        self.up3 = convTranspose(layer[4], layer[3])
        self.conv_u3 = ECABlock(layer[4], layer[3])

        self.up2 = convTranspose(layer[3], layer[2])
        self.conv_u2 = ECABlock(layer[3], layer[2])

        self.up1 = convTranspose(layer[2], layer[1])
        self.conv_u1 = ECAPreBlock(layer[2], layer[1])

        self.up0 = convTranspose(layer[1], layer[0])
        self.conv_u0 = ECAPreBlock(layer[1], layer[0])

        self.conv1 = conv3_3(layer[0], num_classes)

    def forward(self, x):
        x0 = self.conv_d0(x)
        x1 = self.down(x0)
        x1 = self.conv_d1(x1)
        x2 = self.down(x1)
        x2 = self.conv_d2(x2)
        x3 = self.down(x2)
        x3 = self.conv_d3(x3)
        x4 = self.down(x3)

        x4 = self.conv_d4(x4)

        u = self.up3(x4)
        u = torch.cat([x3, u], 1)
        u = self.conv_u3(u)
        u = self.up2(u)
        u = torch.cat([x2, u], 1)
        u = self.conv_u2(u)
        u = self.up1(u)
        u = torch.cat([x1, u], 1)
        u = self.conv_u1(u)
        u = self.up0(u)
        u = torch.cat([x0, u], 1)
        u = self.conv_u0(u)
        u = self.conv1(u)
        return u


class Unet(nn.Module):
    def __init__(self, color_dim=8, num_classes=9):
        super(Unet, self).__init__()
        # layer = [32, 64, 128, 256, 512]
        layer = [64, 128, 256, 512, 1024]
        self.conv_E1 = tow_layer_conv(color_dim, layer[0])
        self.conv_E2 = tow_layer_conv_with_pool(layer[0], layer[1])
        self.conv_E3 = three_layer_conv_with_pool(layer[1], layer[2], layer[2])
        self.conv_E4 = three_layer_conv_with_pool(layer[2], layer[3], layer[3])

        self.conv_E5 = three_layer_conv_with_pool(layer[3], layer[4], layer[4])

        self.conv_u4 = convTranspose(layer[4], layer[3])
        self.conv_p4 = three_layer_conv(layer[4], layer[3], layer[3])

        self.conv_u3 = convTranspose(layer[3], layer[2])
        self.conv_p3 = three_layer_conv(layer[3], layer[2], layer[2])

        self.conv_u2 = convTranspose(layer[2], layer[1])
        self.conv_p2 = tow_layer_conv(layer[2], layer[1])

        self.conv_u1 = convTranspose(layer[1], layer[0])
        self.conv_p1 = tow_layer_conv(layer[1], layer[0])

        self.conv = conv3_3(layer[0], num_classes)

    def forward(self, x):
        x1 = self.conv_E1(x)

        x2 = self.conv_E2(x1)

        x3 = self.conv_E3(x2)

        x4 = self.conv_E4(x3)

        x5 = self.conv_E5(x4)

        x = self.conv_u4(x5)
        x = torch.cat([x4, x], 1)
        x = self.conv_p4(x)

        x = self.conv_u3(x)
        x = torch.cat([x3, x], 1)
        x = self.conv_p3(x)

        x = self.conv_u2(x)
        x = torch.cat([x2, x], 1)
        x = self.conv_p2(x)

        x = self.conv_u1(x)
        x = torch.cat([x1, x], 1)
        x = self.conv_p1(x)

        x = self.conv(x)

        return x


class Segnet(nn.Module):
    def __init__(self, color_dim=13, num_classes=6):
        super(Segnet, self).__init__()
        # layer = [32, 64, 128, 256, 512]
        layer = [64, 128, 256, 512, 1024]
        self.conv_E1 = tow_layer_conv(color_dim, layer[0])
        self.conv_E2 = tow_layer_conv(layer[0], layer[1])
        self.conv_E3 = three_layer_conv(layer[1], layer[2], layer[2])
        self.conv_E4 = three_layer_conv(layer[2], layer[3], layer[3])
        self.conv_E5 = three_layer_conv(layer[3], layer[4], layer[4])

        self.conv_p4 = three_layer_conv(layer[4], layer[3], layer[3])
        self.conv_p3 = three_layer_conv(layer[3], layer[2], layer[2])
        self.conv_p2 = tow_layer_conv(layer[2], layer[1])
        self.conv_p1 = tow_layer_conv(layer[1], layer[0])
        self.conv_p0 = nn.Sequential(
            nn.Conv2d(layer[0], layer[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(layer[0]), nn.ReLU(),
            nn.Conv2d(layer[0], num_classes, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        id = []
        x = self.conv_E1(x)
        x, id1 = f.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id1)

        x = self.conv_E2(x)
        x, id2 = f.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id2)

        x = self.conv_E3(x)
        x, id3 = f.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id3)

        x = self.conv_E4(x)
        x, id4 = f.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id4)

        x = self.conv_E5(x)
        x, id5 = f.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id5)

        x = f.max_unpool2d(x, id[4], kernel_size=(2, 2), stride=(2, 2))
        x = self.conv_p4(x)

        x = f.max_unpool2d(x, id[3], kernel_size=(2, 2), stride=(2, 2))
        x = self.conv_p3(x)

        x = f.max_unpool2d(x, id[2], kernel_size=(2, 2), stride=(2, 2))
        x = self.conv_p2(x)

        x = f.max_unpool2d(x, id[1], kernel_size=(2, 2), stride=(2, 2))
        x = self.conv_p1(x)

        x = f.max_unpool2d(x, id[0], kernel_size=(2, 2), stride=(2, 2))
        x = self.conv_p0(x)

        return x


class UnetPP(nn.Module):
    def __init__(self, color_dim=1, num_classes=2):
        super().__init__()
        layer = [32, 64, 128, 256,512]
        self.conv01 = tow_layer_conv(color_dim, layer[0])
        self.conv02 = tow_layer_conv_with_pool(layer[0], layer[1])
        self.conv03 = three_layer_conv_with_pool(layer[1], layer[2], layer[2])
        self.conv04 = three_layer_conv_with_pool(layer[2], layer[3], layer[3])
        self.conv05 = three_layer_conv_with_pool(layer[3], layer[4], layer[4])

        self.conv11 = tow_layer_conv(layer[1], layer[0])
        self.conv12 = tow_layer_conv(layer[2], layer[1])
        self.conv13 = tow_layer_conv(layer[3], layer[2])
        self.conv14 = tow_layer_conv(layer[4], layer[3])

        self.conv21 = tow_layer_conv(3 * layer[0], layer[0])
        self.conv22 = tow_layer_conv(3 * layer[1], layer[1])
        self.conv23 = tow_layer_conv(3 * layer[2], layer[2])

        self.conv31 = tow_layer_conv(layer[2], layer[0])
        self.conv32 = tow_layer_conv(layer[3], layer[1])

        self.conv41 = tow_layer_conv(5 * layer[0], layer[0])

        self.up_conv11 = convTranspose(layer[1], layer[0])
        self.up_conv12 = convTranspose(layer[2], layer[1])
        self.up_conv13 = convTranspose(layer[3], layer[2])
        self.up_conv14 = convTranspose(layer[4], layer[3])

        self.up_conv21 = convTranspose(layer[1], layer[0])
        self.up_conv22 = convTranspose(layer[2], layer[1])
        self.up_conv23 = convTranspose(layer[3], layer[2])

        self.up_conv31 = convTranspose(layer[1], layer[0])
        self.up_conv32 = convTranspose(layer[2], layer[1])
        self.up_conv41 = convTranspose(layer[1], layer[0])

        self.conv = nn.Conv2d(layer[0], num_classes, kernel_size=(1, 1))

    def forward(self, img):
        c01 = self.conv01(img)
        c02 = self.conv02(c01)
        c11 = self.up_conv11(c02)
        c11 = torch.cat((c01, c11), 1)
        c11 = self.conv11(c11)
        out11 = self.conv(c11)

        c03 = self.conv03(c02)
        c12 = self.up_conv12(c03)
        c12 = torch.cat((c02, c12), 1)
        c12 = self.conv12(c12)
        c21 = self.up_conv21(c12)
        c21 = torch.cat((c01, c11, c21), 1)
        c21 = self.conv21(c21)
        out21 = self.conv(c21)

        c04 = self.conv04(c03)
        c13 = self.up_conv13(c04)
        c13 = torch.cat((c03, c13), 1)
        c13 = self.conv13(c13)
        c22 = self.up_conv22(c13)
        c22 = torch.cat((c02, c12, c22), 1)
        c22 = self.conv22(c22)
        c31 = self.up_conv31(c22)
        c31 = torch.cat((c01, c11, c21, c31), 1)
        c31 = self.conv31(c31)
        out31 = self.conv(c31)

        c05 = self.conv05(c04)
        c14 = self.up_conv14(c05)
        c14 = torch.cat((c04, c14), 1)
        c14 = self.conv14(c14)
        c23 = self.up_conv23(c14)
        c23 = torch.cat((c03, c13, c23), 1)
        c23 = self.conv23(c23)
        c32 = self.up_conv32(c23)
        c32 = torch.cat((c02, c12, c22, c32), 1)
        c32 = self.conv32(c32)
        c41 = self.up_conv41(c32)
        c41 = torch.cat((c01, c11, c21, c31, c41), 1)
        c41 = self.conv41(c41)
        out41 = self.conv(c41)

        return out11 + out21 + out31 + out41


class Unet3P(nn.Module):
    """based on the backbone of VGG16"""

    def __init__(self, color_dim=13, num_classes=9):
        super(Unet3P, self).__init__()
        # layer = [32, 64, 128, 256, 512]
        layer = [64, 128, 256, 512, 1024]
        self.conv_E1 = tow_layer_conv(color_dim, layer[0])
        self.conv_E2 = tow_layer_conv_with_pool(layer[0], layer[1])
        self.conv_E3 = three_layer_conv_with_pool(layer[1], layer[2], layer[2])
        self.conv_E4 = three_layer_conv_with_pool(layer[2], layer[3], layer[3])
        self.conv_E5 = three_layer_conv_with_pool(layer[3], layer[4], layer[4])

        self.conv5_3 = conv3_3(layer[4], layer[0])
        self.conv4_3 = conv3_3(layer[3], layer[0])
        self.conv3_3 = conv3_3(layer[2], layer[0])
        self.conv2_3 = conv3_3(layer[1], layer[0])
        self.conv1_3 = conv3_3(layer[0], layer[0])
        self.conv3 = conv3_3(5 * layer[0], layer[0])

        self.conv1 = conv1_1(5 * layer[0], num_classes)
        self.conv5_1 = conv1_1(layer[4], num_classes)

        self.maxpool4 = nn.MaxPool2d(kernel_size=(4, 4), stride=4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.maxpool8 = nn.MaxPool2d(kernel_size=(8, 8), stride=8)

    def forward(self, img):
        xe1 = self.conv_E1(img)
        xe2 = self.conv_E2(xe1)
        xe3 = self.conv_E3(xe2)
        xe4 = self.conv_E4(xe3)
        xe5 = self.conv_E5(xe4)

        xd41 = f.interpolate(xe5, scale_factor=2, mode='bilinear', align_corners=True)
        xd41 = self.conv5_3(xd41)
        xd42 = self.conv4_3(xe4)
        xd43 = self.maxpool2(xe3)
        xd43 = self.conv3_3(xd43)
        xd44 = self.maxpool4(xe2)
        xd44 = self.conv2_3(xd44)
        xd45 = self.maxpool8(xe1)
        xd45 = self.conv1_3(xd45)
        xd4 = torch.cat([xd41, xd42, xd43, xd44, xd45], 1)
        out4 = self.conv1(xd4)

        xd31 = f.interpolate(xe5, scale_factor=4, mode='bilinear', align_corners=True)
        xd31 = self.conv5_3(xd31)
        xd32 = f.interpolate(xd4, scale_factor=2, mode='bilinear', align_corners=True)
        xd32 = self.conv3(xd32)
        xd33 = self.conv3_3(xe3)
        xd34 = self.maxpool2(xe2)
        xd34 = self.conv2_3(xd34)
        xd35 = self.maxpool4(xe1)
        xd35 = self.conv1_3(xd35)
        xd3 = torch.cat([xd31, xd32, xd33, xd34, xd35], 1)
        out3 = self.conv1(xd3)

        xd21 = f.interpolate(xe5, scale_factor=8, mode='bilinear', align_corners=True)
        xd21 = self.conv5_3(xd21)
        xd22 = f.interpolate(xd4, scale_factor=4, mode='bilinear', align_corners=True)
        xd22 = self.conv3(xd22)
        xd23 = f.interpolate(xd3, scale_factor=2, mode='bilinear', align_corners=True)
        xd23 = self.conv3(xd23)
        xd24 = self.conv2_3(xe2)
        xd25 = self.maxpool2(xe1)
        xd25 = self.conv1_3(xd25)
        xd2 = torch.cat([xd21, xd22, xd23, xd24, xd25], 1)
        out2 = self.conv1(xd2)

        xd11 = f.interpolate(xe5, scale_factor=16, mode='bilinear', align_corners=True)
        xd11 = self.conv5_3(xd11)
        xd12 = f.interpolate(xd4, scale_factor=8, mode='bilinear', align_corners=True)
        xd12 = self.conv3(xd12)
        xd13 = f.interpolate(xd3, scale_factor=4, mode='bilinear', align_corners=True)
        xd13 = self.conv3(xd13)
        xd14 = f.interpolate(xd2, scale_factor=2, mode='bilinear', align_corners=True)
        xd14 = self.conv3(xd14)
        xd15 = self.conv1_3(xe1)
        xd1 = torch.cat([xd11, xd12, xd13, xd14, xd15], 1)
        out1 = self.conv1(xd1)

        return out1


class Unet4P(nn.Module):
    """based on the backbone of VGG16"""

    def __init__(self, color_dim=1, num_classes=2):
        super(Unet4P, self).__init__()
        self.conv_x11 = tow_layer_conv(color_dim, 16)
        self.conv_x12 = tow_layer_conv_with_pool(16, 32)
        self.conv_x13 = three_layer_conv_with_pool(32, 64, 64)
        self.conv_x14 = three_layer_conv_with_pool(64, 128, 128)
        self.conv_x15 = three_layer_conv_with_pool(128, 256, 256)

        self.conv_x21 = tow_layer_conv(32, 16)
        self.conv_x22 = tow_layer_conv(64, 32)
        self.conv_x23 = three_layer_conv(128, 64, 64)
        self.conv_x24 = three_layer_conv(256, 128, 128)

        self.conv_x31 = tow_layer_conv(48, 16)
        self.conv_x32 = tow_layer_conv(96, 32)
        self.conv_x33 = three_layer_conv(192, 64, 64)

        self.conv_x41 = tow_layer_conv(64, 16)
        self.conv_x42 = tow_layer_conv(96, 32)

        self.conv_x51 = tow_layer_conv(80, 16)

        self.up_conv_x21 = convTranspose(32, 16)
        self.up_conv_x22 = convTranspose(64, 32)
        self.up_conv_x23 = convTranspose(128, 64)
        self.up_conv_x24 = convTranspose(256, 128)

        self.conv = conv1_1(16, num_classes)
        self.conv13 = conv3_3(64, 16)
        self.conv14 = conv3_3(128, 32)
        self.conv15 = conv3_3(256, 64)
        self.conv14_41 = conv3_3(128, 16)
        self.conv15_51 = conv3_3(256, 16)

    def forward(self, x):
        x11 = self.conv_x11(x)
        x12 = self.conv_x12(x11)
        x21_1 = self.up_conv_x21(x12)
        x21 = torch.cat((x11, x21_1), 1)
        x21 = self.conv_x21(x21)
        out21 = self.conv(x21)

        x13 = self.conv_x13(x12)
        x22_1 = self.up_conv_x22(x13)
        x22 = torch.cat((x12, x22_1), 1)
        x22 = self.conv_x22(x22)
        x31_1 = self.up_conv_x21(x22)
        x31_2 = f.interpolate(x13, scale_factor=4, mode='bilinear', align_corners=True)
        x31_2 = self.conv13(x31_2)

        x31 = torch.cat([x31_1, x31_2, x21], 1)
        x31 = self.conv_x31(x31)
        out31 = self.conv(x31)

        x14 = self.conv_x14(x13)
        x23_1 = self.up_conv_x23(x14)
        x23 = torch.cat([x13, x23_1], 1)
        x23 = self.conv_x23(x23)
        x32_1 = self.up_conv_x22(x23)
        x32_2 = f.interpolate(x14, scale_factor=4, mode='bilinear', align_corners=True)
        x32_2 = self.conv14(x32_2)
        x32 = torch.cat([x32_1, x22, x32_2], 1)
        x32 = self.conv_x32(x32)
        x41_1 = self.up_conv_x21(x32)
        x41_2 = f.interpolate(x23, scale_factor=4, mode='bilinear', align_corners=True)
        x41_2 = self.conv13(x41_2)
        x41_3 = f.interpolate(x14, scale_factor=8, mode='bilinear', align_corners=True)
        x41_3 = self.conv14_41(x41_3)
        x41 = torch.cat([x41_1, x41_2, x31, x41_3], 1)
        x41 = self.conv_x41(x41)
        out41 = self.conv(x41)

        x15 = self.conv_x15(x14)
        x24_1 = self.up_conv_x24(x15)
        x24 = torch.cat([x24_1, x14], 1)
        x24 = self.conv_x24(x24)
        x33_1 = self.up_conv_x23(x24)
        x33_2 = f.interpolate(x15, scale_factor=4, mode='bilinear', align_corners=True)
        x33_2 = self.conv15(x33_2)
        x33 = torch.cat([x33_1, x23, x33_2], 1)
        x33 = self.conv_x33(x33)
        x42_1 = self.up_conv_x22(x33)
        x42_2 = f.interpolate(x24, scale_factor=4, mode='bilinear', align_corners=True)
        x42_2 = self.conv14(x42_2)
        x42 = torch.cat([x42_1, x32, x42_2], 1)
        x42 = self.conv_x42(x42)
        x51_1 = self.up_conv_x21(x42)
        x51_2 = f.interpolate(x33, scale_factor=4, mode='bilinear', align_corners=True)
        x51_2 = self.conv13(x51_2)
        x51_3 = f.interpolate(x24, scale_factor=8, mode='bilinear', align_corners=True)
        x51_3 = self.conv14_41(x51_3)
        x51_4 = f.interpolate(x15, scale_factor=16, mode='bilinear', align_corners=True)
        x51_4 = self.conv15_51(x51_4)
        x51 = torch.cat([x51_1, x51_2, x51_3, x51_4, x41], 1)
        x51 = self.conv_x51(x51)
        out51 = self.conv(x51)

        return out51 + out41 + out31 + out21


class Unet5P(nn.Module):
    def __init__(self, color_dim=1, num_classes=2):
        super().__init__()
        layer = [32, 64, 128, 256, 512]
        self.conv_x11 = tow_layer_conv(color_dim, layer[0])
        self.conv_x12 = tow_layer_conv_with_pool(layer[0], layer[1])
        self.conv_x13 = three_layer_conv_with_pool(layer[1], layer[2], layer[2])
        self.conv_x14 = three_layer_conv_with_pool(layer[2], layer[3], layer[3])
        self.conv_x15 = three_layer_conv_with_pool(layer[3], layer[4], layer[4])

        self.conv_x21 = tow_layer_conv(layer[2], layer[0])
        self.conv_x22 = tow_layer_conv(layer[2], layer[1])
        self.conv_x23 = tow_layer_conv(layer[3], layer[2])
        self.conv_x24 = tow_layer_conv(layer[3], layer[3])

        self.conv_x31 = tow_layer_conv(layer[2], layer[0])
        self.conv_x32 = tow_layer_conv(layer[2], layer[1])
        self.conv_x33 = tow_layer_conv(layer[2], layer[2])
        self.conv_x41 = tow_layer_conv(layer[2], layer[0])
        self.conv_x42 = tow_layer_conv(layer[1], layer[1])
        self.conv_x51 = tow_layer_conv(layer[2], layer[0])

        self.up_conv_x21 = convTranspose(layer[1], layer[0])
        self.up_conv_x22 = convTranspose(layer[2], layer[1])
        self.up_conv_x23 = convTranspose(layer[3], layer[2])
        self.up_conv_x24 = convTranspose(layer[4], layer[3])

        self.conv = conv1_1(layer[0], num_classes)
        self.conv15 = conv3_3(layer[4], layer[0])
        self.conv24 = conv3_3(layer[3], layer[0])
        self.conv33 = conv3_3(layer[2], layer[0])

    def forward(self, x):
        x11 = self.conv_x11(x)
        x12 = self.conv_x12(x11)
        x13 = self.conv_x13(x12)
        x14 = self.conv_x14(x13)
        x15 = self.conv_x15(x14)

        x24 = self.up_conv_x24(x15)
        x24 = self.conv_x24(x24)

        x23_1 = self.up_conv_x23(x24)
        x23_2 = self.up_conv_x23(x14)
        x23 = torch.cat([x23_1, x23_2], 1)
        x23 = self.conv_x23(x23)

        x22_1 = self.up_conv_x22(x23)
        x22_2 = self.up_conv_x22(x13)
        x22 = torch.cat([x22_1, x22_2], 1)
        x22 = self.conv_x22(x22)

        x21_1 = self.up_conv_x21(x22)
        x21_2 = f.interpolate(x23, scale_factor=4, mode='bilinear', align_corners=True)
        x21_2 = self.conv33(x21_2)
        x21_3 = f.interpolate(x24, scale_factor=8, mode='bilinear', align_corners=True)
        x21_3 = self.conv24(x21_3)
        x21_4 = self.up_conv_x21(x12)
        x21 = torch.cat([x21_1, x21_2, x21_3, x21_4], 1)
        x21 = self.conv_x21(x21)
        out1 = self.conv(x21)

        x33 = self.up_conv_x23(x24)
        x33 = self.conv_x33(x33)

        x32_1 = self.up_conv_x22(x33)
        x32_2 = self.up_conv_x22(x23)
        x32 = torch.cat([x32_1, x32_2], 1)
        x32 = self.conv_x32(x32)

        x31_1 = self.up_conv_x21(x22)
        x31_2 = self.up_conv_x21(x32)
        x31_3 = f.interpolate(x13, scale_factor=4, mode='bilinear', align_corners=True)
        x31_3 = self.conv33(x31_3)
        x31_4 = f.interpolate(x33, scale_factor=4, mode='bilinear', align_corners=True)
        x31_4 = self.conv33(x31_4)
        x31 = torch.cat([x31_1, x31_2, x31_3, x31_4], 1)
        x31 = self.conv_x31(x31)
        out2 = self.conv(x31)

        x42 = self.up_conv_x22(x33)
        x42 = self.conv_x42(x42)

        x41_1 = self.up_conv_x21(x42)
        x41_2 = self.up_conv_x21(x32)
        x41_3 = f.interpolate(x14, scale_factor=8, mode='bilinear', align_corners=True)
        x41_3 = self.conv24(x41_3)
        x41_4 = f.interpolate(x23, scale_factor=4, mode='bilinear', align_corners=True)
        x41_4 = self.conv33(x41_4)
        x41 = torch.cat([x41_1, x41_2, x41_3, x41_4], 1)
        x41 = self.conv_x41(x41)
        out3 = self.conv(x41)

        x51_1 = self.up_conv_x21(x42)
        x51_2 = f.interpolate(x33, scale_factor=4, mode='bilinear', align_corners=True)
        x51_2 = self.conv33(x51_2)
        x51_3 = f.interpolate(x24, scale_factor=8, mode='bilinear', align_corners=True)
        x51_3 = self.conv24(x51_3)
        x51_4 = f.interpolate(x15, scale_factor=16, mode='bilinear', align_corners=True)
        x51_4 = self.conv15(x51_4)
        x51 = torch.cat([x51_1, x51_2, x51_3, x51_4], 1)
        x51 = self.conv_x51(x51)
        out4 = self.conv(x51)

        return out1 + out2 + out3 + out4


class SEUnet(nn.Module):
    def __init__(self, color_dim=13, num_classes=9):
        super().__init__()
        layer = [32, 64, 128, 256, 512]
        # layer = [64, 128, 256, 512, 1024]
        self.down_sample = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv_f1 = SEPreBlock(color_dim, layer[0])
        self.conv_f2 = SEPreBlock(layer[0], layer[1])
        self.conv_f3 = SEBlock(layer[1], layer[2])
        self.conv_f4 = SEBlock(layer[2], layer[3])

        self.conv_f5 = SEBlock(layer[3], layer[4])

        self.up_conv4 = convTranspose(layer[4], layer[3])
        self.up_f4 = SEBlock(layer[4], layer[3])

        self.up_conv3 = convTranspose(layer[3], layer[2])
        self.up_f3 = SEBlock(layer[3], layer[2])

        self.up_conv2 = convTranspose(layer[2], layer[1])
        self.up_f2 = SEPreBlock(layer[2], layer[1])

        self.up_conv1 = convTranspose(layer[1], layer[0])
        self.up_f1 = SEPreBlock(layer[1], layer[0])

        self.output = conv3_3(layer[0], num_classes)

    def forward(self, x):
        c1_1 = self.conv_f1(x)
        c1_2 = self.down_sample(c1_1)

        c2_1 = self.conv_f2(c1_2)
        c2_2 = self.down_sample(c2_1)

        c3_1 = self.conv_f3(c2_2)
        c3_2 = self.down_sample(c3_1)

        c4_1 = self.conv_f4(c3_2)
        c4_2 = self.down_sample(c4_1)

        c5 = self.conv_f5(c4_2)

        u4_1 = self.up_conv4(c5)
        u4 = torch.cat([u4_1, c4_1], 1)
        u4 = self.up_f4(u4)

        u3_1 = self.up_conv3(u4)
        u3 = torch.cat([u3_1, c3_1], 1)
        u3 = self.up_f3(u3)

        u2_1 = self.up_conv2(u3)
        u2 = torch.cat([u2_1, c2_1], 1)
        u2 = self.up_f2(u2)

        u1_1 = self.up_conv1(u2)
        u1 = torch.cat([u1_1, c1_1], 1)
        u1 = self.up_f1(u1)

        return self.output(u1)


if __name__ == '__main__':
    unet_2d = Unet(13, 9)
    x = torch.rand(1, 13, 128, 128)
    x = torch.autograd.Variable(x)
    print(x.size())
    start_time = time.time()
    y = unet_2d(x)
    end_time = time.time()
    print('-' * 50)
    print('run time: %.2f' % (end_time - start_time))
    print(y.size())
