import os
from PIL import Image
import torchvision.transforms as transforms


class ImageData(object):

    def __init__(self, load_size, channels, data_path, ids):

        self.load_size = load_size
        self.channels = channels
        self.ids = ids

        self.data_path = data_path
        file_names = [f for f in os.listdir(data_path)
                       if f.endswith('.jpg')]

        self.file_dict = {}
        for f_name in file_names:
        # 提取屬性：identity, head_pose, side
            fields = f_name.split('.')[0].split('_')
            identity = fields[0]
            head_pose = fields[2]
            side = fields[-1]
            key = '_'.join([identity, head_pose, side])
            if key not in self.file_dict:
                self.file_dict[key] = []
            self.file_dict[key].append(f_name) # f_name是照片的完整名稱

        self.train_images = []
        self.train_angles_r = []
        self.train_labels = []
        self.train_images_t = []
        self.train_angles_g = []

        self.test_images = []
        self.test_angles_r = []
        self.test_labels = []
        self.test_images_t = []
        self.test_angles_g = []

    def image_processing(
        self,
        filename,
        angles_r,
        labels,
        filename_t,
        angles_g
    ):
        
        def _to_image(file_name):
            # 加載圖片
            img = Image.open(file_name).convert("RGB" if self.channels == 3 else "L")
            
            # 定義圖片轉換流水線
            transform = transforms.Compose([
                transforms.Resize((self.load_size, self.load_size)),  # 調整大小
                transforms.ToTensor(),  # 轉換為張量並將像素值歸一化到 [0, 1]
                transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels)  # 將像素值歸一化到 [-1, 1]
            ])
            
            # 應用轉換
            img = transform(img)
            
            return img
        
        image = _to_image(filename)
        image_t = _to_image(filename_t)

        return image, angles_r, labels, image_t, angles_g
    
    def preprocess(self):

        for key, file_list in self.file_dict.items():  # 同一個人不同角度的照片
            if len(file_list) == 1:  # 如果只有一張圖片，跳過
                continue
            
            idx = int(key.split('_')[0])
            flip = 1
            if key.split('_')[-1] == 'R':  # 判斷是否為右眼
                flip = -1
            
            for f_r in file_list: # 對於每個key（ex: 0010_0P_R）的照片
                file_path = os.path.join(self.data_path, f_r)
                h_angle_r = flip * float(f_r.split('_')[-2].split('H')[0]) / 15.0  # 水平方向角度
                v_angle_r = float(f_r.split('_')[-3].split('V')[0]) / 10.0  # 垂直方向角度

                for f_g in file_list:
                    file_path_t = os.path.join(self.data_path, f_g)
                    h_angle_g = flip * float(f_g.split('_')[-2].split('H')[0]) / 15.0
                    v_angle_g = float(f_g.split('_')[-3].split('V')[0]) / 10.0
                    
                    if idx <= self.ids:  # 訓練集
                        self.train_images.append(file_path)
                        self.train_angles_r.append([h_angle_r, v_angle_r])
                        self.train_labels.append(idx - 1)
                        self.train_images_t.append(file_path_t)
                        self.train_angles_g.append([h_angle_g, v_angle_g])
                    else:  # 測試集
                        self.test_images.append(file_path)
                        self.test_angles_r.append([h_angle_r, v_angle_r])
                        self.test_labels.append(idx - 1)
                        self.test_images_t.append(file_path_t)
                        self.test_angles_g.append([h_angle_g, v_angle_g])

        print('\nFinished preprocessing the dataset...')