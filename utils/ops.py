import torch
import torch.nn.functional as F
import numpy as np


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


def conv2d(input_, out_channels, d_h=2, d_w=2, scope='conv_0',
           conv_filters_dim=4, padding='zero', use_bias=True, pad=0):

    k_h = k_w = conv_filters_dim

    # Handle padding
    if pad > 0:
        if padding == 'zero':
            input_ = F.pad(input_, (pad, pad, pad, pad), mode='constant', value=0)
        elif padding == 'reflect':
            input_ = F.pad(input_, (pad, pad, pad, pad), mode='reflect')

    # Apply convolution
    weight = torch.nn.Parameter(
        torch.randn(out_channels, input_.size(1), k_h, k_w) * 0.02
    )  # Mimicking tf.random_normal_initializer(stddev=0.02)
    bias = (
        torch.nn.Parameter(torch.zeros(out_channels)) if use_bias else None
    )  # Mimicking tf.constant_initializer(0)

    conv = F.conv2d(input_, weight, bias, stride=(d_h, d_w), padding=0)

    return conv

def deconv2d(input_, out_channels, d_h=2, d_w=2, scope='deconv_0',
             conv_filters_dim=4, padding='SAME', use_bias=True):
    
    k_h = k_w = conv_filters_dim

    # Calculate output padding if using "SAME" padding
    if padding.upper() == 'SAME':
        pad_h = (input_.size(2) - 1) * d_h + k_h - input_.size(2)
        pad_w = (input_.size(3) - 1) * d_w + k_w - input_.size(3)
        output_padding = (pad_h % d_h, pad_w % d_w)
        padding_mode = (k_h // 2, k_w // 2)  # Implicit SAME padding
    elif padding.upper() == 'VALID':
        output_padding = (0, 0)
        padding_mode = 0
    else:
        raise ValueError("Invalid padding type. Choose 'SAME' or 'VALID'.")

    # Weight initialization
    weight = torch.nn.Parameter(
        torch.randn(input_.size(1), out_channels, k_h, k_w) * 0.02
    )  # Mimicking tf.random_normal_initializer(stddev=0.02)
    bias = (
        torch.nn.Parameter(torch.zeros(out_channels)) if use_bias else None
    )  # Mimicking tf.constant_initializer(0)

    # Transposed convolution
    deconv = F.conv_transpose2d(
        input_,
        weight=weight,
        bias=bias,
        stride=(d_h, d_w),
        padding=padding_mode,
        output_padding=output_padding
    )

    return deconv

def relu(input_):

    return torch.relu(input_)

def lrelu(input_):

    return torch.nn.functional.leaky_relu(input_, negative_slope=0.01)

def tanh(input_):

    return torch.tanh(input_)

def l1_loss(x, y):
    
    loss = torch.mean(torch.abs(x - y))
    return loss

def l2_loss(x, y):

    loss = torch.mean(torch.sum((x - y) ** 2, dim=(1, 2, 3)))
    return loss

def content_loss(endpoints_mixed, content_layers):
   
    loss = 0
    for layer in content_layers:
        # Split the tensor along the batch dimension into two halves
        feat_a, feat_b = torch.split(endpoints_mixed[layer], split_size_or_sections=2, dim=0)
        
        # Compute L2 loss and normalize by the size of the tensor
        size = feat_a.numel()  # Total number of elements in feat_a
        loss += torch.nn.functional.mse_loss(feat_a, feat_b, reduction='sum') * 2 / size

    return loss

def style_loss(endpoints_mixed, style_layers):
    
    loss = 0
    for layer in style_layers:
        # Split the tensor along the batch dimension into two halves
        feat_a, feat_b = torch.split(endpoints_mixed[layer], split_size_or_sections=2, dim=0)

        # Compute the size of the tensor
        size = feat_a.numel()

        # Calculate the Gram matrices for both feature maps
        gram_a = gram(feat_a)
        gram_b = gram(feat_b)

        # Compute L2 loss between the Gram matrices and normalize
        loss += torch.nn.functional.mse_loss(gram_a, gram_b, reduction='sum') * 2 / size

    return loss

def gram(layer):
    
    b, h, w, c = layer.size()
    features = layer.view(b, h * w, c)  # Flatten spatial dimensions (height, width)
    denominator = h * w * c  # Normalize by the total number of elements
    grams = torch.bmm(features.transpose(1, 2), features) / denominator  # Batch matrix multiplication and normalization
    return grams

def angular2cart(angular): #角度座標轉換為笛卡爾座標

    theta = angular[:, 0] / 180.0 * np.pi
    phi = angular[:, 1] / 180.0 * np.pi
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)

    return np.stack([x, y, z], axis=1)

def angular_error(x, y):

    x = angular2cart(x) #換成笛卡爾座標
    y = angular2cart(y) #換成笛卡爾座標

    x_norm = np.sqrt(np.sum(np.square(x), axis=1))
    y_norm = np.sqrt(np.sum(np.square(y), axis=1))

    # 使用 np.clip 函數將 sim 限制在 [−1,1][−1,1] 範圍內，以避免計算反餘弦時出現錯誤
    sim = np.divide(np.sum(np.multiply(x, y), axis=1),
                    np.multiply(x_norm, y_norm))

    sim = np.clip(sim, -1.0, 1.0)

    return np.arccos(sim) * 180.0 / np.pi #計算餘弦相似度對應的角度，並將結果轉換為度數
