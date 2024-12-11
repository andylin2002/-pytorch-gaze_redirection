import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.archs import Discriminator, Generator, vgg_16
from src.data_loader import ImageData
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

import numpy as np
from torchvision.utils import save_image

from utils.ops import l1_loss, content_loss, style_loss, angular_error
from utils.eyes_catch import eyes_catch
from utils.paste import paste

from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.generator = Generator(style_dim=2)
        self.discriminator = Discriminator(params)
        if params.vgg_path:
            vgg_dict = torch.load(params.vgg_path, map_location=torch.device('cpu'), weights_only=True)
            self.vgg_dict = vgg_dict
            print(f"Successfully loaded pretrained weights from {params.vgg_path}")

        self.params = params  # 存儲傳遞的參數
        self.global_step = torch.tensor(0, dtype=torch.int32, requires_grad=False)  # 全局步數
        self.lr = params.lr

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)  # 偏置初始化為 0

        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)

    def data_loader(self):##
        
        hps = self.params

        image_data_class = ImageData(load_size=hps.image_size,
                                     channels=3,
                                     data_path=hps.data_path,
                                     ids=hps.ids)
        image_data_class.preprocess()

        train_dataset_num = len(image_data_class.train_images)
        test_dataset_num = len(image_data_class.test_images)


        train_images = []
        train_angles_r = []
        train_labels = []
        train_images_t = []
        train_angles_g = []

        '''train_data'''
        tqdm.write(f"train dataset number: {train_dataset_num}")
        for each in tqdm(range(train_dataset_num)):
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
        tqdm.write(f"test dataset number: {test_dataset_num}")
        for each in tqdm(range(test_dataset_num)):
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
        test_loader = DataLoader(test_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=8)

        return train_loader, test_loader, train_dataset_num
    
    def adv_loss(self, images_r, images_g):

        hps = self.params

        # 判別器對真實樣本和生成樣本的輸出
        gan_real, reg_real = self.discriminator(images_r)
        gan_fake, reg_fake = self.discriminator(images_g)

        # 生成插值樣本
        eps = torch.rand((hps.batch_size, 1, 1, 1), device=images_r.device)
        interpolated = eps * images_r + (1. - eps) * images_g
        interpolated = interpolated.detach().requires_grad_(True)

        gan_inter, _ = self.discriminator(interpolated)

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
        adv_d_loss = (-torch.mean(gan_real) +
                torch.mean(gan_fake) +
                10.0 * gp)

        # 生成器損失 (Generator Loss)
        adv_g_loss = -torch.mean(gan_fake) 

        # 迴歸損失 (Regression Loss)
        reg_d_loss = F.mse_loss(self.angles_r, reg_real)
        reg_g_loss = F.mse_loss(self.angles_g, reg_fake)

        return adv_d_loss, adv_g_loss, reg_d_loss, reg_g_loss, gp

    def feat_loss(self, image_g, image_t):

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
        inputs = torch.cat([image_g, image_t], dim=0) # shape = [32+32, 3, 64, 64]

        # 加載預訓練的 VGG16 模型
        _, end_points = vgg_16(self, inputs)
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
        c_loss = content_loss(hps, endpoints_mixed, content_layers)
        s_loss = style_loss(hps, endpoints_mixed, style_layers)

        return c_loss, s_loss

    def d_loss_calculator(self, images_r, angles_r, images_t, angles_g):

        images_g = self.generator(images_r, angles_g)

        images_recon = self.generator(images_g, angles_r)

        # reconstruction loss
        self.recon_loss = l1_loss(images_r, images_recon)

        # content loss and style loss
        self.c_loss, self.s_loss = self.feat_loss(images_g, images_t)

        # regression losses and adversarial losses
        (self.adv_d_loss, self.adv_g_loss, self.reg_d_loss,
        self.reg_g_loss, self.gp) = self.adv_loss(images_r, images_g)

        return self.adv_d_loss + 5.0 * self.reg_d_loss

    def g_loss_calculator(self, images_r, angles_r, images_t, angles_g):

        images_g = self.generator(images_r, angles_g)

        images_recon = self.generator(images_g, angles_r)

        # reconstruction loss
        self.recon_loss = l1_loss(images_r, images_recon)

        # content loss and style loss
        self.c_loss, self.s_loss = self.feat_loss(images_g, images_t)

        # regression losses and adversarial losses
        (self.adv_d_loss, self.adv_g_loss, self.reg_d_loss,
        self.reg_g_loss, self.gp) = self.adv_loss(images_r, images_g)

        return (self.adv_g_loss + 3.0 * self.reg_g_loss +  # self.adv_g_loss 定義已加負號
                                    self.recon_loss +
                                    self.s_loss + self.c_loss)
    
    def optimizer(self, lr, model):

        hps = self.params

        if hps.optimizer == 'sgd':
            return optim.SGD(model.parameters(), lr=lr)
        if hps.optimizer == 'adam':
            return optim.Adam(model.parameters(),
                            lr=lr,
                            betas=(hps.adam_beta1, hps.adam_beta2))
        raise AttributeError("attribute 'optimizer' is not assigned!")

    def add_optimizer(self, learning_rate):

        # 創建優化器
        g_op = self.optimizer(learning_rate, self.generator)
        d_op = self.optimizer(learning_rate, self.discriminator)

        return d_op, g_op
    
    def add_summary(self, writer: SummaryWriter, step: int):

        # 記錄標量
        writer.add_scalar('Loss/recon_loss', self.recon_loss.item(), step)
        writer.add_scalar('Loss/adv_g_loss', self.adv_g_loss.item(), step)
        writer.add_scalar('Loss/adv_d_loss', self.adv_d_loss.item(), step)
        writer.add_scalar('Loss/reg_d_loss', self.reg_d_loss.item(), step)
        writer.add_scalar('Loss/reg_g_loss', self.reg_g_loss.item(), step)
        writer.add_scalar('Metrics/gp', self.gp.item(), step)
        writer.add_scalar('Learning_Rate', self.lr, step)
        writer.add_scalar('Loss/c_loss', self.c_loss.item(), step)
        writer.add_scalar('Loss/s_loss', self.s_loss.item(), step)

        # 記錄影像
        real_images = (self.x_r + 1) / 2.0
        #fake_images = torch.clamp((self.x_g + 1) / 2.0, 0., 1.)
        #recon_images = torch.clamp((self.x_recon + 1) / 2.0, 0., 1.)
        #valid_images = torch.clamp((self.x_valid_r + 1) / 2.0, 0., 1.)

        writer.add_images('Images/real', real_images, step)
        #writer.add_images('Images/fake', fake_images, step)
        #writer.add_images('Images/recon', recon_images, step)
        #writer.add_images('Images/x_test', valid_images, step)
    
    def train(self, conti = False):

        hps = self.params

        current_model_path = os.path.join(hps.log_dir, "current_model.ckpt")
        if conti:
            checkpoint = torch.load(current_model_path)
            self.generator.load_state_dict(checkpoint['current_generator'])
            self.discriminator.load_state_dict(checkpoint['current_discriminator'])
            print(f". ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.Loaded previous model. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.")

        best_model_loss = float('inf')

        (train_iter, test_iter, train_size) = self.data_loader() #加載訓練、驗證和測試數據集的迭代器

        num_epoch = hps.epochs
        batch_size = hps.batch_size
        learning_rate = hps.lr

        num_iter = train_size // batch_size # num_iter = 1102

        # 日誌與模型路徑
        summary_dir = os.path.join(hps.log_dir, 'summary')
        summary_writer = SummaryWriter(log_dir=summary_dir)

        # 設定 GPU 動態記憶體增長
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            # PyTorch 不需要顯式設定動態記憶體增長，它會自動優化 GPU 記憶體使用
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        self.device = device

        # to device
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        
        try:
            for epoch in range(num_epoch):
                print(f"Epoch: {epoch+1}/{num_epoch}")

                # 動態調整學習率
                if epoch >= num_epoch // 2:
                    min_lr = 0.1 * hps.lr  # 最小學習率
                    learning_rate = max(min_lr, (2.0 - 2.0 * epoch / num_epoch) * hps.lr)

                # update operations for generator and discriminator
                self.d_op, self.g_op = self.add_optimizer(learning_rate)

                for it in tqdm(range(num_iter)):
                    #print(f"batch: {it}/{num_iter}")
                    # 從 DataLoader 提取批次數據
                    train_batch = [t.to(device) for t in next(iter(train_iter))]
                    test_batch = [t.to(device) for t in next(iter(test_iter))]

                    # 解包訓練數據
                    self.x_r, self.angles_r, self.labels, self.x_t, self.angles_g = train_batch
                    '''
                    self.x_r: torch.Size([32, 3, 64, 64]),
                    self.angles_r: torch.Size([32, 2]),
                    self.labels: torch.Size([32]),
                    self.x_t: torch.Size([32, 3, 64, 64]),
                    self.angles_g: torch.Size([32, 2])
                    '''

                    # 解包測試數據
                    self.x_test_r, self.angles_test_r, self.labels_test, self.x_test_t, self.angles_test_g = test_batch
                    '''
                    self.x_test_r: torch.Size([32, 3, 64, 64]),
                    self.angles_test_r: torch.Size([32, 2]),
                    self.labels_test: torch.Size([32]),
                    self.x_test_t: torch.Size([32, 3, 64, 64]),
                    self.angles_test_g: torch.Size([32, 2])
                    '''

                    # 訓練 Discriminator
                    self.d_op.zero_grad()

                    # 暫時固定 Generator 的參數
                    for param in self.generator.parameters():
                        param.requires_grad = False

                    d_loss = self.d_loss_calculator(self.x_r, self.angles_r, self.x_t, self.angles_g)

                    #print("train discriminator...")
                    d_loss.backward()
                    self.d_op.step()

                    # 解鎖 Generator 的參數
                    for param in self.generator.parameters():
                        param.requires_grad = True

                    # 訓練 Generator (每 5 步執行一次)
                    if it % 5 == 0:

                        self.g_op.zero_grad()

                        # 暫時固定 Discriminator 的參數
                        for param in self.discriminator.parameters():
                            param.requires_grad = False

                        g_loss = self.g_loss_calculator(self.x_r, self.angles_r, self.x_t, self.angles_g)
                        #print(f"generator loss: {g_loss:<15.2f}, discriminator loss: {d_loss:<15.2f}")

                        #print("train generator...")
                        g_loss.backward()
                        self.g_op.step()

                        # 解鎖 Discriminator 的參數
                        for param in self.discriminator.parameters():
                            param.requires_grad = True

                    
                    # 記錄摘要和保存模型
                    if it % hps.summary_steps == 0:
                        self.global_step = epoch * num_iter + it

                        d_test_loss = self.d_loss_calculator(self.x_test_r, self.angles_test_r, self.x_test_t, self.angles_test_g)
                        transformed_d_test_loss = torch.exp(d_test_loss / 10)
                        g_test_loss = self.g_loss_calculator(self.x_test_r, self.angles_test_r, self.x_test_t, self.angles_test_g)
                        transformed_g_test_loss = torch.exp(g_test_loss / 10)
                        tqdm.write(f"generator test loss: {transformed_g_test_loss:<10.2f}, discriminator test loss: {transformed_d_test_loss:<10.2f}")

                        #定義比較模型學習好壞的標準
                        challenger_loss = (transformed_g_test_loss + transformed_d_test_loss * (transformed_g_test_loss + transformed_d_test_loss)) # g + d * (g + d)
                        if challenger_loss < best_model_loss:
                            best_model_loss = challenger_loss
                            tqdm.write(f". ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.New lowest loss at step: {self.global_step}. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.")

                        # 保存模型權重
                        torch.save({
                                'current_generator': self.generator.state_dict(),
                                'current_discriminator': self.discriminator.state_dict()
                            }, current_model_path)

                        # 使用自定義的 add_summary 函式
                        self.add_summary(summary_writer, self.global_step)

        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            torch.save(self.state_dict(), os.path.join("log", "model.ckpt"))

        finally:
            summary_writer.close()

    def eval(self):

        hps = self.params

        checkpoint_path = os.path.join(hps.log_dir, 'current_model.ckpt')
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['current_generator'])

        eval_dir = os.path.join(hps.log_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)

        # 設定 GPU 動態記憶體增長
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            # PyTorch 不需要顯式設定動態記憶體增長，它會自動優化 GPU 記憶體使用
        else:
            device = torch.device("cpu")
        #print(f"Using device: {device}")
        
        for file_name in os.listdir(hps.client_pictures_dir): # for each picture
            each_eval_dir = os.path.join(eval_dir, f'eval_{file_name}')
            os.makedirs(each_eval_dir, exist_ok=True)

            picture_eyes_patch, eyes_position, size = eyes_catch(hps, file_name) # picture_eyes_patch.shape = [eyes' number, 3, 64, 64]

            gaze_angles = torch.zeros(len(picture_eyes_patch), 2)

            with torch.no_grad():  # 禁用梯度計算
                generated_image = self.generator(picture_eyes_patch, gaze_angles) # generated_image.shape = [number, 3, 64, 64]
            
            paste(hps, file_name, generated_image, eyes_position, size)
            
            for i, new_img in enumerate(generated_image):
                if i % 2 == 0: # left eye
                    file_name = os.path.join(each_eval_dir, f'generated_{i // 2}_L.png')
                    save_image(new_img, file_name)
                else: # right eye
                    file_name = os.path.join(each_eval_dir, f'generated_{i // 2}_R.png')
                    save_image(new_img, file_name)

        print("Evaluation finished.")
        