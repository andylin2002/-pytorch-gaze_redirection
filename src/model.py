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

import numpy as np
from torchvision.utils import save_image

from utils.ops import l1_loss, content_loss, style_loss, angular_error

import torchvision.models as models

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
        self.x_r, self.angles_r, self.labels, self.x_t, self.angles_g = train_batch
        print(f"shape :\nself.x_r: {self.x_r.shape},\n self.angles_r: {self.angles_r.shape},\n self.labels: {self.labels.shape},\n self.x_t: {self.x_t.shape},\n self.angles_g: {self.angles_g.shape}\n")

        # 解包驗證數據
        self.x_valid_r, self.angles_valid_r, self.labels_valid, self.x_valid_t, self.angles_valid_g = valid_batch
        print(f"shape :\nself.x_valid_r: {self.x_valid_r.shape},\n self.angles_valid_r: {self.angles_valid_r.shape},\n self.labels_valid: {self.labels_valid.shape},\n self.x_valid_t: {self.x_valid_t.shape},\n self.angles_valid_g: {self.angles_valid_g.shape}\n")

        # 解包測試數據
        self.x_test_r, self.angles_test_r, self.labels_test, self.x_test_t, self.angles_test_g = test_batch
        print(f"shape :\nself.x_test_r: {self.x_test_r.shape},\n self.angles_test_r: {self.angles_test_r.shape},\n self.labels_test: {self.labels_test.shape},\n self.x_test_t: {self.x_test_t.shape},\n self.angles_test_g: {self.angles_test_g.shape}\n")

        self.x_g = generator(self.x_r, self.angles_g)
        self.x_recon = generator(self.x_g, self.angles_r)

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
        print(f"train dataset number: {train_dataset_num}")
        print(f"test dataset number: {test_dataset_num}")

        '''train_data'''

        train_images = []
        train_angles_r = []
        train_labels = []
        train_images_t = []
        train_angles_g = []

        for each in range(train_dataset_num):
            image_data_class.train_images[each], image_data_class.train_angles_r[each], image_data_class.train_labels[each], image_data_class.train_images_t[each], image_data_class.train_angles_g[each] = image_data_class.image_processing(
                image_data_class.train_images[each],
                image_data_class.train_angles_r[each],
                image_data_class.train_labels[each],
                image_data_class.train_images_t[each],
                image_data_class.train_angles_g[each]
            )

        train_images = torch.stack(image_data_class.train_images) if isinstance(image_data_class.train_images[0], torch.Tensor) else torch.tensor(image_data_class.train_images, dtype=torch.float32)
        train_angles_r = torch.tensor(image_data_class.train_angles_r, dtype=torch.float32)
        train_labels = torch.tensor(image_data_class.train_labels, dtype=torch.float32)
        train_images_t = torch.stack(image_data_class.train_images_t) if isinstance(image_data_class.train_images_t[0], torch.Tensor) else torch.tensor(image_data_class.train_images_t, dtype=torch.float32)
        train_angles_g = torch.tensor(image_data_class.train_angles_g, dtype=torch.float32)

        train_dataset = TensorDataset(
            train_images,
            train_angles_r,
            train_labels,
            train_images_t,
            train_angles_g
        )

        '''test_data'''
        for each in range(test_dataset_num):
            image_data_class.test_images[each], image_data_class.test_angles_r[each], image_data_class.test_labels[each], image_data_class.test_images_t[each], image_data_class.test_angles_g[each] = image_data_class.image_processing(
                image_data_class.test_images[each],
                image_data_class.test_angles_r[each],
                image_data_class.test_labels[each],
                image_data_class.test_images_t[each],
                image_data_class.test_angles_g[each]
            )

        test_images = torch.stack(image_data_class.test_images) if isinstance(image_data_class.test_images[0], torch.Tensor) else torch.tensor(image_data_class.test_images, dtype=torch.float32)
        test_angles_r = torch.tensor(image_data_class.test_angles_r, dtype=torch.float32)
        test_labels = torch.tensor(image_data_class.test_labels, dtype=torch.float32)
        test_images_t = torch.stack(image_data_class.test_images_t) if isinstance(image_data_class.test_images_t[0], torch.Tensor) else torch.tensor(image_data_class.test_images_t, dtype=torch.float32)
        test_angles_g = torch.tensor(image_data_class.test_angles_g, dtype=torch.float32)

        test_dataset = TensorDataset(
            test_images,
            test_angles_r,
            test_labels,
            test_images_t,
            test_angles_g
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

        hps = self.params
        
        # 定義 VGG 模型和需要的層名稱
        content_layers = ["conv5"]  # conv5_3
        style_layers = [
            "conv1",   # conv1_2
            "conv2",   # conv2_2
            "conv3",  # conv3_3
            "conv4"   # conv4_3
        ]

        # 拼接輸入：將 x_g 和 x_t 合併在一起
        inputs = torch.cat([self.x_g, self.x_t], dim=0) # shape = [32+32, 3, 64, 64]

        # 加載預訓練的 VGG16 模型
        _, end_points = vgg_16(inputs, hps, pretrained=True)
        '''
        # 禁止更新 VGG 的權重
        for param in vgg.parameters():
            param.requires_grad = False
        '''
        # 通過 VGG 網絡，收集中間層的輸出
        endpoints_mixed = {}
        for layer in content_layers + style_layers:
            endpoints_mixed[layer] = end_points[layer]

        # 計算內容損失和風格損失
        c_loss = content_loss(endpoints_mixed, content_layers)
        s_loss = style_loss(endpoints_mixed, style_layers)

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

        hps = self.params

        num_epoch = hps.epochs
        train_size = self.train_size
        batch_size = hps.batch_size
        learning_rate = hps.lr

        num_iter = train_size // batch_size

        # 日誌與模型路徑
        summary_dir = os.path.join(hps.log_dir, 'summary')
        summary_writer = SummaryWriter(log_dir=summary_dir)

        model_path = os.path.join(hps.log_dir, 'model.ckpt')

        # 設定 GPU 動態記憶體增長
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            # PyTorch 不需要顯式設定動態記憶體增長，它會自動優化 GPU 記憶體使用
        else:
            device = torch.device("cpu")
        #print(f"Using device: {device}")

        optimizer_d, optimizer_g, loss_d, loss_g = self.add_optimizer()

        try:
            for epoch in range(num_epoch):
                print(f"Epoch: {epoch+1}/{num_epoch}")

                # 動態調整學習率
                if epoch >= num_epoch // 2:
                    learning_rate = (2.0 - 2.0 * epoch / num_epoch) * hps.lr
                    for param_group in optimizer_d.param_groups:
                        param_group['lr'] = learning_rate
                    for param_group in optimizer_g.param_groups:
                        param_group['lr'] = learning_rate

                for it in range(num_iter):
                    # 訓練 Discriminator
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()

                    # 訓練 Generator (每 5 步執行一次)
                    if it % 5 == 0:
                        optimizer_g.zero_grad()
                        loss_g.backward()
                        optimizer_g.step()

                    # 記錄摘要和保存模型
                    if it % hps.summary_steps == 0:
                        self.global_step = epoch * num_iter + it

                        # 使用自定義的 add_summary 函式
                        self.add_summary(summary_writer, self.global_step)

                        # 保存模型權重
                        torch.save(self.model.state_dict(), f"{model_path}_{self.global_step}.pth")

        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            torch.save(self.model.state_dict(), f"{model_path}_final.pth")

        finally:
            summary_writer.close()



    def eval(self):

        hps = self.params

        checkpoint_path = os.path.join(hps.log_dir, 'model.ckpt')
        self.generator.load_state_dict(torch.load(checkpoint_path))

        # 設定 GPU 動態記憶體增長
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            # PyTorch 不需要顯式設定動態記憶體增長，它會自動優化 GPU 記憶體使用
        else:
            device = torch.device("cpu")
        #print(f"Using device: {device}")

        imgs_dir = os.path.join(hps.log_dir, 'eval')
        os.makedirs(imgs_dir, exist_ok=True)

        tar_dir = os.path.join(imgs_dir, 'targets')
        gene_dir = os.path.join(imgs_dir, 'genes')
        real_dir = os.path.join(imgs_dir, 'reals')

        os.makedirs(tar_dir, exist_ok=True)
        os.makedirs(gene_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)

        with torch.no_grad():
            for i, (real_imgs, target_imgs, angles_r, angles_g) in enumerate(self.test_loader):
                real_imgs, target_imgs, angles_r, angles_g = (
                    real_imgs.to(hps.device),
                    target_imgs.to(hps.device),
                    angles_r.to(hps.device),
                    angles_g.to(hps.device),
                )

                # Generate fake images
                fake_imgs = self.generator(real_imgs, angles_g)

                # Calculate angular errors
                a_t = angles_g.cpu().numpy() * np.array([15, 10])
                a_r = angles_r.cpu().numpy() * np.array([15, 10])
                delta = angular_error(a_t, a_r)

                # Save images
                for j in range(real_imgs.size(0)):
                    save_image(target_imgs[j], os.path.join(
                        tar_dir, f"{i}_{j}_{delta[j]:.3f}_H{a_t[j][0]}_V{a_t[j][1]}.jpg"))

                    save_image(fake_imgs[j], os.path.join(
                        gene_dir, f"{i}_{j}_{delta[j]:.3f}_H{a_t[j][0]}_V{a_t[j][1]}.jpg"))

                    save_image(real_imgs[j], os.path.join(
                        real_dir, f"{i}_{j}_{delta[j]:.3f}_H{a_t[j][0]}_V{a_t[j][1]}.jpg"))

        print("Evaluation finished.")
        