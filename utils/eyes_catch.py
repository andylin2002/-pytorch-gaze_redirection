import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import torch

import dlib
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def eyes_catch(hps, file_name):

    channels = 3

    # 專案的根目錄路徑
    ROOT_DIR = "/home/andy/AILab/AIfinal/pytorch_gaze_redirection-master/"

    # 訓練/驗證用的資料目錄
    DATA_PATH = os.path.join(ROOT_DIR, "client_pictures/")

    # 模型資料目錄
    MODEL_PATH = os.path.join(ROOT_DIR, "utils/eyes_catch_model")

    MODEL_FILE_PATH = os.path.join(MODEL_PATH, "shape_predictor_68_face_landmarks.dat")

    # 測試用圖像
    #TEST_IMAGE = os.path.join(ROOT_DIR, "2004andy2.JPG")

    #64乘64輸出圖
    #OUTPUT_PATH = os.path.join(ROOT_DIR, "eyespatch_dataset")

    '''----------face detection----------'''

    # 使用 dlib 自帶的 frontal_face_detector 作為人臉偵測器
    face_detector = dlib.get_frontal_face_detector()

    # 遍歷資料夾內的每個圖片
    valid_extensions = {".jpg"}  # 支援的圖片格式
    file_path = os.path.join(DATA_PATH, file_name)
    
    # 檢查文件是否為圖片
    if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in valid_extensions:
        try:
            # 載入圖像檔
            image = Image.open(file_path)
            
            # 把 PIL.Image 物件轉換成 numpy ndarray
            img = np.array(image)
            dets = face_detector(img, 0) # 因為測試的圖像己經很大了, 因此我們不啟動upsampling
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # dets的元素個數即為偵測到臉的個數
    '''print("Number of faces detected: {}".format(len(dets)))'''

    '''----------crop eyes patch----------'''

    predictor = dlib.shape_predictor(MODEL_FILE_PATH)  # 使用您設定的模型路徑

    picture_eyes_patch = []

    eyes_position = []

    size = []

    # 遍歷每個偵測到的人臉
    for i, d in enumerate(dets):
        '''
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}"
            .format(i, d.left(), d.top(), d.right(), d.bottom()))
        '''
        # 使用 Dlib 的特徵點偵測器
        landmarks = predictor(img, d)

        # 獲取左眼和右眼的特徵點位置
        left_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        # 計算眼睛區域的邊界框
        left_eye_x1 = min([p[0] for p in left_eye_points])
        left_eye_y1 = min([p[1] for p in left_eye_points])
        left_eye_x2 = max([p[0] for p in left_eye_points])
        left_eye_y2 = max([p[1] for p in left_eye_points])

        right_eye_x1 = min([p[0] for p in right_eye_points])
        right_eye_y1 = min([p[1] for p in right_eye_points])
        right_eye_x2 = max([p[0] for p in right_eye_points])
        right_eye_y2 = max([p[1] for p in right_eye_points])

        # 計算正方形的邊長，以長邊為主，並放大1.5倍
        left_eye_width = left_eye_x2 - left_eye_x1
        left_eye_height = left_eye_y2 - left_eye_y1
        left_eye_size = int(1.5 * max(left_eye_width, left_eye_height))

        right_eye_width = right_eye_x2 - right_eye_x1
        right_eye_height = right_eye_y2 - right_eye_y1
        right_eye_size = int(1.5 * max(right_eye_width, right_eye_height))

        # 更新左眼和右眼的邊界框，使其為放大1.5倍的正方形，並保持中心不變
        left_eye_center_x = (left_eye_x1 + left_eye_x2) // 2
        left_eye_center_y = (left_eye_y1 + left_eye_y2) // 2
        left_eye_x1 = left_eye_center_x - left_eye_size // 2
        left_eye_x2 = left_eye_center_x + left_eye_size // 2
        left_eye_y1 = left_eye_center_y - left_eye_size // 2
        left_eye_y2 = left_eye_center_y + left_eye_size // 2

        right_eye_center_x = (right_eye_x1 + right_eye_x2) // 2
        right_eye_center_y = (right_eye_y1 + right_eye_y2) // 2
        right_eye_x1 = right_eye_center_x - right_eye_size // 2
        right_eye_x2 = right_eye_center_x + right_eye_size // 2
        right_eye_y1 = right_eye_center_y - right_eye_size // 2
        right_eye_y2 = right_eye_center_y + right_eye_size // 2

        # 假設我們已經從 Dlib 得到左眼和右眼的邊界框座標
        left_eye = image.crop((left_eye_x1, left_eye_y1, left_eye_x2, left_eye_y2))
        right_eye = image.crop((right_eye_x1, right_eye_y1, right_eye_x2, right_eye_y2))
        size.append(left_eye_size)
        size.append(right_eye_size)

        transform = transforms.Compose([
                transforms.Resize((hps.image_size, hps.image_size)),  # 調整大小
                transforms.ToTensor(),  # 轉換為張量並將像素值歸一化到 [0, 1]
                transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels)  # 將像素值歸一化到 [-1, 1]
            ])
        
        left_eye_tensor = transform(left_eye)
        right_eye_tensor = transform(right_eye)

        picture_eyes_patch.append(left_eye_tensor)
        picture_eyes_patch.append(right_eye_tensor)

        eyes_position.append([left_eye_x1, left_eye_y1])
        eyes_position.append([right_eye_x1, right_eye_y1])

    # 將列表轉換為 Tensor
    picture_eyes_patch = torch.stack(picture_eyes_patch)

    return picture_eyes_patch, eyes_position, size



