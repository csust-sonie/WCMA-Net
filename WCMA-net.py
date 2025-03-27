from WCM import *
from  MSAM  import *
import torch.nn as nn
import warnings
from torchvision import models
warnings.filterwarnings('ignore')

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, scale_factor=2, group_size=4, use_scope=False):
        super(PixelShuffleUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.group_size = group_size
        assert in_channels >= group_size and in_channels % group_size == 0

        out_channels = 2 * group_size * scale_factor ** 2

        self.offset_conv = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset_conv, std=0.001)
        if use_scope:
            self.scope_conv = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope_conv, val=0.)

        self.register_buffer('initial_pos', self._initialize_position())

    def _initialize_position(self):
        h = torch.arange((-self.scale_factor + 1) / 2, (self.scale_factor - 1) / 2 + 1) / self.scale_factor
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.group_size, 1).reshape(1, -1, 1, 1)

    def sample_coordinates(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale_factor).view(
            B, 2, -1, self.scale_factor * H, self.scale_factor * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.group_size, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale_factor * H, self.scale_factor * W)

    def forward(self, x):
        offset = self.offset_conv(x) * 0.25 + self.initial_pos
        return self.sample_coordinates(x, offset)

class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet101FeatureExtractor, self).__init__()
        resnet = models.resnet101(pretrained=pretrained)

        # 使用 ResNet 的多个层
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256通道
        self.layer2 = resnet.layer2  # 512通道
        self.layer3 = resnet.layer3  # 1024通道
        self.layer4 = resnet.layer4  # 2048通道

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # 这里通道数为 1024
        f4 = self.layer4(x)  # 这里通道数为 2048
        return f4  # 返回最后一层特征
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class WCMA_Net(nn.Module):
    def __init__(self, pretrained=True):
        super(WCMA_Net, self).__init__()

        # 使用 ResNet101 作为特征提取器，提取最后一层特征
        self.resnet = ResNet101FeatureExtractor(pretrained=pretrained)


        self.wcm= WCM(2048, 2048)
        self.msam = MSAM(in_channels=2048)

        self.conv= nn.Sequential(
            nn.Conv2d(2048, 2048, 1, 1, 0),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        # SEBlock 用于计算自适应权重
        self.se_block = SEBlock(2048)
        # DySample_UP 上采样模块，前面添加卷积层来调整通道数
        self.reduce_channels1 = nn.Conv2d(2048, 1024, kernel_size=1)
        self.upsample1 = PixelShuffleUpsample(in_channels=1024, scale_factor=2)  # [1, 2048, 8, 8] -> [1, 1024, 16, 16]

        self.reduce_channels2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.upsample2 = PixelShuffleUpsample(in_channels=512, scale_factor=2)  # [1, 1024, 16, 16] -> [1, 512, 32, 32]

        self.reduce_channels3 = nn.Conv2d(512, 256, kernel_size=1)
        self.upsample3 = PixelShuffleUpsample(in_channels=256, scale_factor=2)  # [1, 512, 32, 32] -> [1, 256, 64, 64]

        self.reduce_channels4 = nn.Conv2d(256, 64, kernel_size=1)
        self.upsample4 = PixelShuffleUpsample(in_channels=64, scale_factor=2)  # [1, 256, 64, 64] -> [1, 64, 128, 128]

        # 注意此处的通道数
        self.reduce_channels5 = nn.Conv2d(64, 4, kernel_size=1)
        self.upsample5 = PixelShuffleUpsample(in_channels=4, scale_factor=2)  # [1, 4, 128, 128] -> [1, 4, 256, 256]

        # 最后再将通道数从 4 变为 1
        self.final_conv = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x):

        # 获取最后一层特
        res_f4 = self.resnet(x)
        wcm=self.wcm(res_f4)
        msam=self.msam(res_f4)


        # 使用 SEBlock 计算自适应权重
        combined_out = wcm + msam
        weight = self.se_block(combined_out)

        # 使用自适应权重调整两个模块的输出
        adaptive_msam = weight * msam
        adaptive_wcm = (1 - weight) * wcm


        fused_out = adaptive_msam+adaptive_wcm
        # print(fused_out.shape)

        fused_out=self.conv(fused_out)
        # print(fused_out.shape)



        # 逐步上采样并手动调整通道数
        fused_out = self.reduce_channels1(fused_out)
        fused_out = self.upsample1(fused_out)  # [1, 2048, 8, 8] -> [1, 1024, 16, 16]

        fused_out = self.reduce_channels2(fused_out)
        fused_out = self.upsample2(fused_out)  # [1, 1024, 16, 16] -> [1, 512, 32, 32]

        fused_out = self.reduce_channels3(fused_out)
        fused_out = self.upsample3(fused_out)  # [1, 512, 32, 32] -> [1, 256, 64, 64]

        fused_out = self.reduce_channels4(fused_out)
        fused_out = self.upsample4(fused_out)  # [1, 256, 64, 64] -> [1, 64, 128, 128]

        fused_out = self.reduce_channels5(fused_out)
        fused_out = self.upsample5(fused_out)  # [1, 4, 128, 128] -> [1, 4, 256, 256]

        # 最终将通道数从 4 降到 1
        fused_out = self.final_conv(fused_out)  # [1, 4, 256, 256] -> [1, 1, 256, 256]

        return fused_out


















