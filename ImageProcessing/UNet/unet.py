from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DownSampling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSampling, self).__init__()
        if bilinear is True:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x2.size()[3]

        # padding_left, padding_right, padding_top padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x1, x2])
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1,
                    num_classes: int = 2,
                    bilinear: bool = False,
                    base_c: int = 64):
        super(UNet, self).__init__()
        self.input = DoubleConv(in_channels, base_c)
        self.down1 = DownSampling(base_c, base_c * 2)
        self.down2 = DownSampling(base_c * 2, base_c * 4)
        self.down3 = DownSampling(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = DownSampling(base_c * 8, base_c * 16 // factor)
        self.up1 = UpSampling(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = UpSampling(base_c * 8 // factor, base_c * 4 // factor, bilinear)
        self.up3 = UpSampling(base_c * 4 // factor, base_c * 2 // factor, bilinear)
        self.up4 = UpSampling(base_c * 2 // factor, base_c, bilinear)
        self.output = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 =self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.output(x)

        return {"out": logits}
