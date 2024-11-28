import torch
import torch.nn as nn
import torch.nn.functional as F

def discriminator(params, x_init):

    layers = 5  # 固定為 5 層卷積
    channel = 64  # 初始通道數
    image_size = params['image_size']  # 從 params 中獲取輸入圖片大小

    # 構建卷積層序列
    conv_layers = []
    conv_layers.append(
        nn.Conv2d(3, channel, kernel_size=4, stride=2, padding=1, bias=True)  # conv_0
    )
    conv_layers.append(nn.LeakyReLU(negative_slope=0.2))  # 激活函數

    for i in range(1, layers):
        conv_layers.append(
            nn.Conv2d(channel, channel * 2, kernel_size=4, stride=2, padding=1, bias=True)  # conv_1 ~ conv_(layers-1)
        )
        conv_layers.append(nn.LeakyReLU(negative_slope=0.2))  # 激活函數
        channel *= 2

    # 卷積層組合為 nn.Sequential
    conv_net = nn.Sequential(*conv_layers)

    # 計算最後輸出層的 kernel_size
    filter_size = image_size // (2 ** layers)

    # 對抗輸出層
    conv_gan = nn.Conv2d(channel, 1, kernel_size=filter_size, stride=1, padding=1, bias=False)

    # 目標回歸輸出層
    conv_reg = nn.Conv2d(channel, 2, kernel_size=filter_size, stride=1, padding=0, bias=False)

    # 前向傳播
    x = conv_net(x_init)
    x_gan = conv_gan(x)  # 對抗輸出
    x_reg = conv_reg(x)  # 回歸輸出
    x_reg = x_reg.view(x_reg.size(0), -1)  # 展平成 [batch_size, 2]

    return x_gan, x_reg

def generator(input_, angles):

    # 確定 channel 初始值
    channel = 64
    style_dim = angles.size(-1)  # 獲取 angles 的最後一維度作為 style_dim

    # Reshape 和 tile angles
    angles_reshaped = angles.view(-1, style_dim, 1, 1)  # 形狀改為 [batch_size, style_dim, 1, 1]
    angles_tiled = angles_reshaped.expand(-1, -1, input_.size(2), input_.size(3))  # 複製到空間維度匹配 input_

    # 在通道維度上拼接
    x = torch.cat([input_, angles_tiled], dim=1)  # 通道方向拼接，dim=1 表示通道維

    # Define the generator model
    layers = []

    # Input layer
    layers.append(nn.Conv2d(x.size(1), channel, kernel_size=7, stride=1, padding=3, bias=False))
    x = layers[-1](x)
    x = instance_norm(x)
    x = F.relu(x)

    # Encoder
    for i in range(2):
        layers.append(nn.Conv2d(channel, channel * 2, kernel_size=4, stride=2, padding=1, bias=False))
        x = layers[-1](x)
        x = instance_norm(x)
        x = F.relu(x)
        channel *= 2

    # Bottleneck (Residual blocks)
    for i in range(6):
        # Residual block part A
        conv_a = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        x_a = conv_a(x)
        x_a = instance_norm(x_a)
        x_a = F.relu(x_a)

             # Residual block part B
        conv_b = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        x_b = conv_b(x_a)
        x_b = instance_norm(x_b)

        x = x + x_b

    # Decoder
    for i in range(2):
        layers.append(nn.ConvTranspose2d(channel, channel // 2, kernel_size=4, stride=2, padding=1, bias=False))
        x = layers[-1](x)
        x = instance_norm(x)
        x = F.relu(x)
        channel //= 2

    # Output layer
    layers.append(nn.Conv2d(channel, 3, kernel_size=7, stride=1, padding=3, bias=False))
    x = layers[-1](x)
    x = torch.tanh(x)

    return x

def vgg_16(inputs, scope='vgg_16'):

    end_points = {}

    # conv1
    net = F.relu(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)(inputs))
    end_points['conv1_1'] = net
    net = F.relu(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv1_2'] = net
    net = F.max_pool2d(net, kernel_size=2, stride=2)
    end_points['pool1'] = net

    # conv2
    net = F.relu(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv2_1'] = net
    net = F.relu(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv2_2'] = net
    net = F.max_pool2d(net, kernel_size=2, stride=2)
    end_points['pool2'] = net

    # conv3
    net = F.relu(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv3_1'] = net
    net = F.relu(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv3_2'] = net
    net = F.relu(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv3_3'] = net
    net = F.max_pool2d(net, kernel_size=2, stride=2)
    end_points['pool3'] = net

    # conv4
    net = F.relu(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv4_1'] = net
    net = F.relu(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv4_2'] = net
    net = F.relu(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv4_3'] = net
    net = F.max_pool2d(net, kernel_size=2, stride=2)
    end_points['pool4'] = net

    # conv5
    net = F.relu(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv5_1'] = net
    net = F.relu(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv5_2'] = net
    net = F.relu(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)(net))
    end_points['conv5_3'] = net
    net = F.max_pool2d(net, kernel_size=2, stride=2)
    end_points['pool5'] = net

    return net, end_points












def instance_norm(x, eps=1e-5):
    """
    Instance Normalization.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    eps : float
        Small value to prevent division by zero.

    Returns
    -------
    x : torch.Tensor
        Normalized tensor.
    """
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + eps)
