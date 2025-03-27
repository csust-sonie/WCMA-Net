import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
nonlinearity = partial(F.relu, inplace=True)

def BNReLU(num_features):
    return nn.Sequential(
        nn.BatchNorm2d(num_features),
        nn.ReLU()
    )
class MultiScalePoolingBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiScalePoolingBlock, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool2 = nn.AdaptiveAvgPool2d((3, 3))
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        self.pool4 = nn.AdaptiveAvgPool2d((6, 6))

        # 使用膨胀卷积，提升细节提取
        self.dilate1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2, groups=in_channels)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3, groups=in_channels)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        pool_1 = self.pool1(x).view(b, c, -1)
        pool_2 = self.pool2(x).view(b, c, -1)
        pool_3 = self.pool3(x).view(b, c, -1)
        pool_4 = self.pool4(x).view(b, c, -1)

        pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)

        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(dilate3_out)))

        cnn_out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        cnn_out = cnn_out.view(b, c, -1)

        out = torch.cat([pool_cat, cnn_out], -1)
        out = out.permute(0, 2, 1)
        return out

class MSAM(nn.Module):
    def __init__(self, in_channels=512, ratio=2):
        super(MSAM, self).__init__()
        self.in_channels = in_channels
        self.key_channels = in_channels // ratio
        self.value_channels = in_channels // ratio

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1),
            BNReLU(self.key_channels),
        )
        self.f_query = self.f_key
        self. f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.value_channels, out_channels=self.in_channels, kernel_size=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        # 多尺度特征池化模块
        self.multi_scale_pool_v = MultiScalePoolingBlock(self.key_channels)
        self.multi_scale_pool_k = MultiScalePoolingBlock(self.key_channels)
        nn.init.constant_(self.W[0].weight, 0)
        nn.init.constant_(self.W[0].bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        x_v = self.f_value(x)
        value = self.multi_scale_pool_v(x_v)

        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        x_k = self.f_key(x)
        key = self.multi_scale_pool_k(x_k)
        key = key.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        return context




