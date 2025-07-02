import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, Conv):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2),
            Conv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, Conv, Pool):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            Pool(2),
            DoubleConv(in_channels, out_channels, Conv)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, Conv, mode):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=mode)
        self.conv = DoubleConv(in_channels, out_channels, Conv)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet_base(nn.Module):
    def __init__(self, n_channels, chs=(32, 64, 128, 256, 512, 256, 128, 64, 32), dimensions='2D'):
        super(UNet_base, self).__init__()
        if dimensions == '2D':
            Conv = nn.Conv2d
            Pool = nn.MaxPool2d
            mode = 'bilinear'
        elif dimensions == '3D':
            Conv = nn.Conv3d
            Pool = nn.MaxPool3d
            mode = 'trilinear'
        else:
            assert "dimensions should be 2D or 3D"

        self.inc = DoubleConv(n_channels, chs[0], Conv)
        self.down1 = Down(chs[0], chs[1], Conv, Pool)
        self.down2 = Down(chs[1], chs[2], Conv, Pool)
        self.down3 = Down(chs[2], chs[3], Conv, Pool)
        self.down4 = Down(chs[3], chs[4], Conv, Pool)
        self.up1 = Up(chs[4] + chs[3], chs[5], Conv, mode)
        self.up2 = Up(chs[5] + chs[2], chs[6], Conv, mode)
        self.up3 = Up(chs[6] + chs[1], chs[7], Conv, mode)
        self.up4 = Up(chs[7] + chs[0], chs[8], Conv, mode)

        self.__init_weight(Conv)

    def __init_weight(self, Conv):
        for m in self.modules():
            if isinstance(m, Conv):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        return x9, x8, x7, x6, x5