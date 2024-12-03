from utils.ops import relu, conv2d, lrelu, instance_norm, deconv2d, tanh
from types import SimpleNamespace
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.layers = 5
        self.channel = 64
        self.image_size = params.image_size

        # 第一層卷積
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.channel, kernel_size=4, stride=2, padding=1)

        # 中間層卷積
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=self.channel * (2 ** i), out_channels=self.channel * (2 ** (i + 1)),
                      kernel_size=4, stride=2, padding=1)
            for i in range(self.layers - 1)
        ])

        # 輸出層 (GAN)
        final_filter_size = int(self.image_size / (2 ** self.layers))
        self.conv_gan = nn.Conv2d(in_channels=self.channel * (2 ** (self.layers - 1)), out_channels=1,
                                  kernel_size=final_filter_size, stride=1, padding=0)

        # 輸出層 (回歸)
        self.conv_reg = nn.Conv2d(in_channels=self.channel * (2 ** (self.layers - 1)), out_channels=2,
                                  kernel_size=final_filter_size, stride=1, padding=0)

    def forward(self, x_init):
        # 第一層卷積
        x = self.conv1(x_init)
        x = F.leaky_relu(x, negative_slope=0.2)  # 使用 PyTorch 的 Leaky ReLU

        # 堆疊中間層
        for conv in self.conv_layers:
            x = conv(x)
            x = F.leaky_relu(x, negative_slope=0.2)

        # GAN 輸出層
        x_gan = self.conv_gan(x)

        # 回歸輸出層
        x_reg = self.conv_reg(x)
        x_reg = x_reg.view(x_reg.size(0), -1)  # 平坦化輸出

        return x_gan, x_reg
    
    def named_parameters_with_prefix(self, prefix='discriminator'):
        for name, param in self.named_parameters():
            yield f"{prefix}.{name}", param


# 測試程式碼
params = SimpleNamespace(image_size=64)
discriminator = Discriminator(params)
input_ = torch.randn(32, 3, 64, 64)
x_gan, x_reg = discriminator(input_)
for name, param in discriminator.named_parameters_with_prefix():
    print(f"Name: {name}, Shape: {param.shape}")
