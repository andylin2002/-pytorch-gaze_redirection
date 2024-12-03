import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops import relu, conv2d, lrelu, instance_norm, deconv2d, tanh

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
        x = F.leaky_relu(self.conv1(x_init), negative_slope=0.2)

        # 堆疊中間層
        for conv in self.conv_layers:
            x = F.leaky_relu(conv(x), negative_slope=0.2)

        # GAN 輸出層
        x_gan = self.conv_gan(x)

        # 回歸輸出層
        x_reg = self.conv_reg(x) # [32, 2, 1, 1]
        x_reg = x_reg.view(x_reg.size(0), -1)  # 平坦化輸出 [32, 2]

        return x_gan, x_reg
    
    def named_parameters_with_prefix(self, prefix='discriminator'):
        for name, param in self.named_parameters():
            yield f"{prefix}.{name}", param

class Generator(nn.Module):
    def __init__(self, style_dim=2):
        super(Generator, self).__init__()
        self.style_dim = style_dim

        # Input layer
        self.input_conv = nn.Conv2d(3 + style_dim, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.input_norm = nn.InstanceNorm2d(64, affine=False)

        # Encoder layers
        self.encoder_convs = nn.ModuleList([
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        ])
        self.encoder_norms = nn.ModuleList([
            nn.InstanceNorm2d(128, affine=False),
            nn.InstanceNorm2d(256, affine=False)
        ])

        # Bottleneck layers
        self.bottleneck_blocks = nn.ModuleList([
            self._residual_block(256) for _ in range(6)
        ])

        # Decoder layers
        self.decoder_convs = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        ])
        self.decoder_norms = nn.ModuleList([
            nn.InstanceNorm2d(128, affine=False),
            nn.InstanceNorm2d(64, affine=False)
        ])

        # Output layer
        self.output_conv = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False)

    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=False)
        )

    def forward(self, input_, angles):
        # Reshape and tile angles
        angles_reshaped = angles.view(-1, self.style_dim, 1, 1)
        angles_tiled = angles_reshaped.expand(-1, self.style_dim, input_.shape[2], input_.shape[3])
        x = torch.cat([input_, angles_tiled], dim=1) # torch.Size([32, 5, 64, 64])
        # Input layer
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.relu(x) # torch.Size([32, 64, 64, 64])

        # Encoder
        for conv, norm in zip(self.encoder_convs, self.encoder_norms):
            x = conv(x)
            x = norm(x)
            x = F.relu(x)
            # torch.Size([32, 128, 32, 32])
            # torch.Size([32, 256, 16, 16])

        # Bottleneck
        for block in self.bottleneck_blocks:
            residual = x
            x = block(x)
            x = x + residual
            # torch.Size([32, 256, 16, 16])
            # torch.Size([32, 256, 16, 16])
            # torch.Size([32, 256, 16, 16])
            # torch.Size([32, 256, 16, 16])
            # torch.Size([32, 256, 16, 16])
            # torch.Size([32, 256, 16, 16])
            

        # Decoder
        for deconv, norm in zip(self.decoder_convs, self.decoder_norms):
            x = deconv(x)
            x = norm(x)
            x = F.relu(x)
            # torch.Size([32, 128, 32, 32])
            # torch.Size([32, 64, 64, 64])

        # Output layer
        x = self.output_conv(x)
        x = torch.tanh(x) # torch.Size([32, 3, 64, 64])
        return x

    def named_parameters_with_prefix(self, prefix='generator'):
        for name, param in self.named_parameters():
            yield f"{prefix}.{name}", param

def vgg_16(inputs, hps, pretrained = True):

    if pretrained:
        pretrained_path = hps.vgg_path

    end_points = {}

    def conv_block(in_channels, out_channels, num_layers, layer_scope):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=False))
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

    # 模型的字典存放所有層
    model_dict = nn.ModuleDict({
        'conv1': conv1,
        'pool1': pool1,
        'conv2': conv2,
        'pool2': pool2,
        'conv3': conv3,
        'pool3': pool3,
        'conv4': conv4,
        'pool4': pool4,
        'conv5': conv5,
        'pool5': pool5
    })

    # 如果提供了預訓練權重的路徑，則載入
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=True)
        try:
            model_dict.load_state_dict(state_dict, strict=False)  # 若權重名稱不完全匹配，允許部分載入
            print(f"Successfully loaded pretrained weights from {pretrained_path}")
        except RuntimeError as e:
            print(f"Error loading pretrained weights: {e}")

    # Forward pass
    x = model_dict['conv1'](inputs)
    end_points[scope1] = x
    x = model_dict['pool1'](x)
    end_points[pool1_scope] = x

    x = model_dict['conv2'](x)
    end_points[scope2] = x
    x = model_dict['pool2'](x)
    end_points[pool2_scope] = x

    x = model_dict['conv3'](x)
    end_points[scope3] = x
    x = model_dict['pool3'](x)
    end_points[pool3_scope] = x

    x = model_dict['conv4'](x)
    end_points[scope4] = x
    x = model_dict['pool4'](x)
    end_points[pool4_scope] = x

    x = model_dict['conv5'](x)
    end_points[scope5] = x
    x = model_dict['pool5'](x)
    end_points[pool5_scope] = x

    return x, end_points







'''
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
               conv_filters_dim=7, use_bias=False, pad=3)
    x = instance_norm(x, scope='in_input')
    x = relu(x)
    # x.shape = torch.Size([32, 64, 64, 64])

    # Encoder
    for i in range(2):
        x = conv2d(x, out_channels=2 * channel, d_h=2, d_w=2, scope=f'conv2d_{i}',
                   conv_filters_dim=3, use_bias=False, pad=1)
        x = instance_norm(x, scope=f'in_conv_{i}')
        x = relu(x)
        channel = 2 * channel
        # x.shape = torch.Size([32, 128, 32, 32])
        # x.shape = torch.Size([32, 256, 16, 16])

    # Bottleneck
    for i in range(6):
        x_a = conv2d(x, out_channels=channel, d_h=1, d_w=1, scope=f'conv_res_a_{i}',
                     conv_filters_dim=3, use_bias=False, pad=1)
        x_a = instance_norm(x_a, scope=f'in_res_a_{i}')
        x_a = relu(x_a)
        x_b = conv2d(x, out_channels=channel, d_h=1, d_w=1, scope=f'conv_res_b_{i}',
                     conv_filters_dim=3, use_bias=False, pad=1)
        x_b = instance_norm(x_b, scope=f'in_res_b_{i}')
        x = x + x_b
        # x.shape = torch.Size([32, 256, 16, 16])
        # x.shape = torch.Size([32, 256, 16, 16])
        # x.shape = torch.Size([32, 256, 16, 16])
        # x.shape = torch.Size([32, 256, 16, 16])
        # x.shape = torch.Size([32, 256, 16, 16])
        # x.shape = orch.Size([32, 256, 16, 16])

    # Decoder
    for i in range(2):
        x = deconv2d(x, out_channels=int(channel / 2), conv_filters_dim=3, d_h=2, d_w=2,
                     use_bias=False, scope=f'deconv_{i}')
        x = instance_norm(x, scope=f'in_decon_{i}')
        x = relu(x)
        channel = int(channel / 2)
        # x.shape = torch.Size([32, 128, 32, 32])
        # x.shape = torch.Size([32, 64, 64, 64])

    # Output layer
    x = conv2d(x, out_channels=3, conv_filters_dim=7, d_h=1, d_w=1, pad=3,
               use_bias=False, scope='output')
    x = tanh(x)
    # x.shape = torch.Size([32, 3, 64, 64])

    print(x.shape)###

    return x
'''