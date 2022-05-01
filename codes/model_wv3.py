# ---------------------------------------------------------------
# Copyright (c) 2022, Zhi-Xuan Chen, Cheng Jin, Xiao Wu, Liang-Jian Deng
# All rights reserved.
#
# This work is licensed under GNU Affero General Public License
# v3.0 International To view a copy of this license, see the
# LICENSE file.
#
# This file is running on WorldView-3 dataset. For other dataset
# (i.e., QuickBird), please change the corresponding
# inputs.
# ---------------------------------------------------------------

import torch
import torch.nn as nn

# --------------------------------SpanConv Block -----------------------------------#
class SpanConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpanConv, self).__init__()
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size

        self.point_wise_1 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)

        self.depth_wise_1 = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=(kernel_size - 1) // 2,
                                    groups=out_channels,
                                    bias=True)

        self.point_wise_2 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)

        self.depth_wise_2 = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=(kernel_size - 1) // 2,
                                    groups=out_channels,
                                    bias=True)


    def forward(self, x):  #
        out_tmp_1 = self.point_wise_1(x)  #
        out_tmp_1 = self.depth_wise_1(out_tmp_1)  #

        out_tmp_2 = self.point_wise_2(x)  #
        out_tmp_2 = self.depth_wise_2(out_tmp_2)  #

        out = out_tmp_1 + out_tmp_2

        return out

# --------------------------------Belly Block -----------------------------------#
class Belly_Block(nn.Module):
    def __init__(self,in_planes):
        super(Belly_Block, self).__init__()
        self.conv1=SpanConv(in_planes,in_planes,3)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=SpanConv(in_planes,in_planes,3)

    def forward(self,x):
        res=self.conv1(x)
        res=self.relu1(res)
        res=self.conv2(res)
        return res


class LightNet(nn.Module):
    def __init__(self):
        super(LightNet, self).__init__()
        self.head_conv=nn.Sequential(
            SpanConv(9, 9, 3),
            SpanConv(9, 20, 3),
            # nn.Conv2d(9,32,3,1,1),
            SpanConv(20,32,3),
            nn.ReLU(inplace=True)
        )

        self.belly_conv = nn.Sequential(
            Belly_Block(32),
            Belly_Block(32)

        )

        self.tail_conv=nn.Sequential(
            # nn.Conv2d(32,8,3,1,1),
            SpanConv(32, 16, 3),
            SpanConv(16, 8, 3),
            SpanConv(8, 8, 3)
        )

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,pan,lms):
        x=torch.cat([pan,lms],1)
        x=self.head_conv(x)
        x = self.belly_conv(x)
        x=self.tail_conv(x)
        sr=lms+x
        return sr


if __name__ == '__main__':
    from torchsummary import summary
    N=LightNet()
    summary(N,[(1,64,64),(8,64,64)],device='cpu')

