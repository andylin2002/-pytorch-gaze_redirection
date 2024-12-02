import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops import relu, conv2d, lrelu, instance_norm, deconv2d, tanh

def discriminator(params, x_init):

    layers = 5
    channel = 64
    image_size = params.image_size

    # 64 3 -> 32 64 -> 16 128 -> 8 256 -> 4 512 -> 2 1024

    # 第一層卷積
    x = conv2d(x_init, out_channels=channel, conv_filters_dim=4, d_h=2, d_w=2,
               scope='conv_0', pad=1, use_bias=True)
    x = lrelu(x)

    # 堆疊中間層
    for i in range(1, layers):
        x = conv2d(x, out_channels=channel * 2, conv_filters_dim=4, d_h=2, d_w=2,
                   scope=f'conv_{i}', pad=1, use_bias=True)
        x = lrelu(x)
        channel = channel * 2

    # 計算輸出層的卷積核大小
    filter_size = int(image_size / (2 ** layers))

    # GAN 輸出層
    x_gan = conv2d(x, out_channels=1, conv_filters_dim=filter_size, d_h=1, d_w=1,
                   pad=0, scope='conv_logit_gan', use_bias=False)

    # 回歸輸出層
    x_reg = conv2d(x, out_channels=2, conv_filters_dim=filter_size, d_h=1, d_w=1,
                   pad=0, scope='conv_logit_reg', use_bias=False)
    x_reg = x_reg.view(x_reg.size(0), -1)  # 平坦化輸出

    return x_gan, x_reg

def generator(input_, angles):

    channel = 64
    style_dim = angles.shape[-1] # style_dim = 2

    # 重塑並複製角度特徵
    angles_reshaped = angles.view(-1, style_dim, 1, 1) # angles_reshaped.shape = [32, 2, 1, 1]
    angles_tiled = angles_reshaped.expand(-1, style_dim, input_.shape[2], input_.shape[3]) # angles_tiled.shape = [32, 2, 64, 64]
    x = torch.cat([input_, angles_tiled], dim=1) # x.shape = [32, 5, 64, 64]

    # 定義生成器的卷積結構
    # Input layer
    x = conv2d(x, out_channels=channel, d_h=1, d_w=1, scope='conv2d_input', 
               use_bias=False, pad=3, conv_filters_dim=7)
    x = instance_norm(x, scope='in_input')
    x = relu(x)

    # Encoder
    for i in range(2):
        x = conv2d(x, out_channels=2 * channel, d_h=2, d_w=2, scope=f'conv2d_{i}',
                   use_bias=False, pad=1, conv_filters_dim=4)
        x = instance_norm(x, scope=f'in_conv_{i}')
        x = relu(x)
        channel = 2 * channel

    # Bottleneck
    for i in range(6):
        x_a = conv2d(x, out_channels=channel, conv_filters_dim=3, d_h=1, d_w=1,
                     pad=1, use_bias=False, scope=f'conv_res_a_{i}')
        x_a = instance_norm(x_a, scope=f'in_res_a_{i}')
        x_a = relu(x_a)
        x_b = conv2d(x_a, out_channels=channel, conv_filters_dim=3, d_h=1, d_w=1,
                     pad=1, use_bias=False, scope=f'conv_res_b_{i}')
        x_b = instance_norm(x_b, scope=f'in_res_b_{i}')
        x = x + x_b

    # Decoder
    for i in range(2):
        x = deconv2d(x, out_channels=int(channel / 2), conv_filters_dim=4, d_h=2, d_w=2,
                     use_bias=False, scope=f'deconv_{i}')
        x = instance_norm(x, scope=f'in_decon_{i}')
        x = relu(x)
        channel = int(channel / 2)

    # Output layer
    x = conv2d(x, out_channels=3, conv_filters_dim=7, d_h=1, d_w=1, pad=3,
               use_bias=False, scope='output')
    x = tanh(x)

    return x

def vgg_16(inputs):

    end_points = {}

    def conv_block(in_channels, out_channels, num_layers, layer_scope):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers), f"{layer_scope}"

    def max_pool(scope):
        return nn.MaxPool2d(kernel_size=2, stride=2), f"{scope}"

    # VGG-16 blocks
    conv1, scope1 = conv_block(3, 64, 2, 'conv1')
    pool1, pool1_scope = max_pool('pool1')

    conv2, scope2 = conv_block(64, 128, 2, 'conv2')
    pool2, pool2_scope = max_pool('pool2')

    conv3, scope3 = conv_block(128, 256, 3, 'conv3')
    pool3, pool3_scope = max_pool('pool3')

    conv4, scope4 = conv_block(256, 512, 3, 'conv4')
    pool4, pool4_scope = max_pool('pool4')

    conv5, scope5 = conv_block(512, 512, 3, 'conv5')
    pool5, pool5_scope = max_pool('pool5')

    # Forward pass
    x = conv1(inputs)
    end_points[scope1] = x
    x = pool1(x)
    end_points[pool1_scope] = x

    x = conv2(x)
    end_points[scope2] = x
    x = pool2(x)
    end_points[pool2_scope] = x

    x = conv3(x)
    end_points[scope3] = x
    x = pool3(x)
    end_points[pool3_scope] = x

    x = conv4(x)
    end_points[scope4] = x
    x = pool4(x)
    end_points[pool4_scope] = x

    x = conv5(x)
    end_points[scope5] = x
    x = pool5(x)
    end_points[pool5_scope] = x

    return x, end_points