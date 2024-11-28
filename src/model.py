import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.archs import discriminator, generator, vgg_16
from src.data_loader import ImageData
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

class Model(object):
    def __init__(self, params):
        self.params = params  # 存儲傳遞的參數
        self.global_step = torch.tensor(0, dtype=torch.int32, requires_grad=False)  # 全局步數
        self.lr = None  # PyTorch 不需要明確的 placeholder，學習率直接用變數傳遞

        (self.train_iter, self.valid_iter,
         self.test_iter, self.train_size) = self.data_loader() #加載訓練、驗證和測試數據集的迭代器
        
        # 從 DataLoader 提取批次數據
        train_batch = next(iter(self.train_iter))
        valid_batch = next(iter(self.valid_iter))
        test_batch = next(iter(self.test_iter))

        # 解包訓練數據
        self.x_r = train_batch['x_r']
        self.angles_r = train_batch['angles_r']
        self.labels = train_batch['labels']
        self.x_t = train_batch['x_t']
        self.angles_g = train_batch['angles_g']

        # 解包驗證數據
        self.x_valid_r = valid_batch['x_r']
        self.angles_valid_r = valid_batch['angles_r']
        self.labels_valid = valid_batch['labels']
        self.x_valid_t = valid_batch['x_t']
        self.angles_valid_g = valid_batch['angles_g']

        # 解包測試數據
        self.x_test_r = test_batch['x_r']
        self.angles_test_r = test_batch['angles_r']
        self.labels_test = test_batch['labels']
        self.x_test_t = test_batch['x_t']
        self.angles_test_g = test_batch['angles_g']

        self.x_g = generator(self.x_r, self.angles_g)
        self.x_recon = generator(self.x_g, self.angles_r, reuse=True)

        self.angles_valid_g = (torch.rand(params.batch_size, 2) * 2.0) - 1.0

        self.x_valid_g = generator(self.x_valid_r, self.angles_valid_g)

        # reconstruction loss
        self.recon_loss = l1_loss(self.x_r, self.x_recon)

        # content loss and style loss
        self.c_loss, self.s_loss = self.feat_loss()

        # regression losses and adversarial losses
        (self.d_loss, self.g_loss, self.reg_d_loss,
         self.reg_g_loss, self.gp) = self.adv_loss()

        # update operations for generator and discriminator
        self.d_op, self.g_op = self.add_optimizer()

        # adding summaries
        self.summary = self.add_summary()

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)  # 偏置初始化為 0

        generator.apply(init_weights)
        discriminator.apply(init_weights)

    def data_loader(self):##
        
        hps = self.params

        image_data_class = ImageData(load_size=hps.image_size,
                                     channels=3,
                                     data_path=hps.data_path,
                                     ids=hps.ids)
        image_data_class.preprocess()

        train_dataset_num = len(image_data_class.train_images)
        test_dataset_num = len(image_data_class.test_images)

        train_dataset = TensorDataset(
            torch.tensor(image_data_class.train_images),
            torch.tensor(image_data_class.train_angles_r),
            torch.tensor(image_data_class.train_labels),
            torch.tensor(image_data_class.train_images_t),
            torch.tensor(image_data_class.train_angles_g)
        )

        test_dataset = TensorDataset(
            torch.tensor(image_data_class.test_images),
            torch.tensor(image_data_class.test_angles_r),
            torch.tensor(image_data_class.test_labels),
            torch.tensor(image_data_class.test_images_t),
            torch.tensor(image_data_class.test_angles_g)
        )

        train_loader = DataLoader(train_dataset, batch_size=hps.batch_size, shuffle=True, num_workers=8)
        valid_loader = DataLoader(test_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=8)

        return train_loader, valid_loader, test_loader, train_dataset_num
    
    def adv_loss(self):

        hps = self.params

        # 判別器對真實樣本和生成樣本的輸出
        gan_real, reg_real = discriminator(hps, self.x_r)
        gan_fake, reg_fake = discriminator(hps, self.x_g)

        # 生成插值樣本
        eps = torch.rand((hps.batch_size, 1, 1, 1), device=self.x_r.device)
        interpolated = eps * self.x_r + (1. - eps) * self.x_g
        interpolated.requires_grad_(True)

        gan_inter, _ = discriminator(hps, interpolated)

        # 計算梯度懲罰（Gradient Penalty）
        grad = torch.autograd.grad(
            outputs=gan_inter,
            inputs=interpolated,
            grad_outputs=torch.ones_like(gan_inter),
            create_graph=True,
            retain_graph=True
        )[0]

        slopes = torch.sqrt(torch.sum(grad**2, dim=[1, 2, 3]))
        gp = torch.mean((slopes - 1)**2)

        # 判別器損失 (Discriminator Loss)
        d_loss = (-torch.mean(gan_real) +
                torch.mean(gan_fake) +
                10.0 * gp)

        # 生成器損失 (Generator Loss)
        g_loss = -torch.mean(gan_fake)

        # 迴歸損失 (Regression Loss)
        reg_loss_d = F.mse_loss(self.angles_r, reg_real)
        reg_loss_g = F.mse_loss(self.angles_g, reg_fake)

        return d_loss, g_loss, reg_loss_d, reg_loss_g, gp

    def feat_loss(self):
        
        # 定義 VGG 模型和需要的層名稱
        content_layers = ["features.28"]  # conv5_3
        style_layers = [
            "features.3",   # conv1_2
            "features.8",   # conv2_2
            "features.15",  # conv3_3
            "features.22"   # conv4_3
        ]

        # 加載預訓練的 VGG16 模型
        vgg = vgg_16(pretrained=True).features.to(self.x_g.device).eval()

        # 禁止更新 VGG 的權重
        for param in vgg.parameters():
            param.requires_grad = False

        # 拼接輸入：將 x_g 和 x_t 合併在一起
        inputs = torch.cat([self.x_g, self.x_t], dim=0)

        # 通過 VGG 網絡，收集中間層的輸出
        endpoints_mixed = {}
        x = inputs
        for name, layer in vgg._modules.items():
            x = layer(x)
            if f"features.{name}" in content_layers + style_layers:
                endpoints_mixed[f"features.{name}"] = x

        # 計算內容損失和風格損失
        c_loss = self.content_loss(endpoints_mixed, content_layers)
        s_loss = self.style_loss(endpoints_mixed, style_layers)

        return c_loss, s_loss
    
    def optimizer(self, lr):

        hps = self.params

        if hps.optimizer == 'sgd':
            return optim.SGD(self.parameters(), lr=lr)
        if hps.optimizer == 'adam':
            return optim.Adam(self.parameters(),
                            lr=lr,
                            betas=(hps.adam_beta1, hps.adam_beta2))
        raise AttributeError("attribute 'optimizer' is not assigned!")

    def add_optimizer(self):

        # 獲取模型參數
        g_vars = [p for n, p in self.named_parameters() if 'generator' in n]
        d_vars = [p for n, p in self.named_parameters() if 'discriminator' in n]

        # 創建優化器
        g_op = self.optimizer(self.lr)
        d_op = self.optimizer(self.lr)

        # 計算損失
        g_loss = (self.g_loss + 5.0 * self.reg_g_loss +
                50.0 * self.recon_loss +
                100.0 * self.s_loss + 100.0 * self.c_loss)
        d_loss = self.d_loss + 5.0 * self.reg_d_loss

        # 設置優化器的參數範圍
        g_op.param_groups = [{'params': g_vars}]
        d_op.param_groups = [{'params': d_vars}]

        return d_op, g_op, d_loss, g_loss
    
    def add_summary(self, writer: SummaryWriter, step: int):

        # 記錄標量
        writer.add_scalar('Loss/recon_loss', self.recon_loss.item(), step)
        writer.add_scalar('Loss/g_loss', self.g_loss.item(), step)
        writer.add_scalar('Loss/d_loss', self.d_loss.item(), step)
        writer.add_scalar('Loss/reg_d_loss', self.reg_d_loss.item(), step)
        writer.add_scalar('Loss/reg_g_loss', self.reg_g_loss.item(), step)
        writer.add_scalar('Metrics/gp', self.gp.item(), step)
        writer.add_scalar('Learning_Rate', self.lr, step)
        writer.add_scalar('Loss/c_loss', self.c_loss.item(), step)
        writer.add_scalar('Loss/s_loss', self.s_loss.item(), step)

        # 記錄影像
        real_images = (self.x_r + 1) / 2.0
        fake_images = torch.clamp((self.x_g + 1) / 2.0, 0., 1.)
        recon_images = torch.clamp((self.x_recon + 1) / 2.0, 0., 1.)
        valid_images = torch.clamp((self.x_valid_r + 1) / 2.0, 0., 1.)
        valid_fake_images = torch.clamp((self.x_valid_g + 1) / 2.0, 0., 1.)

        writer.add_images('Images/real', real_images, step)
        writer.add_images('Images/fake', fake_images, step)
        writer.add_images('Images/recon', recon_images, step)
        writer.add_images('Images/x_test', valid_images, step)
        writer.add_images('Images/x_test_fake', valid_fake_images, step)
    
    def train(self):



    def eval(self):



    
        