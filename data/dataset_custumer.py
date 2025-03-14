import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from PIL import Image
from torchvision import transforms
import random



class CustomTrainDataset(Dataset):
    def __init__(self, data_path, classname, img_size, args):
        
        self.root_dir = os.path.join(data_path, 'train', 'good')
        self.anomaly_dir = os.path.join(data_path, 'train', 'anomaly')
        self.mask_dir = os.path.join(data_path, 'train', 'mask')
        
        # 加载正常和异常图像路径
        self.normal_images = sorted(glob.glob(os.path.join(self.root_dir, "*.jpg")))
        self.anomaly_images = sorted(glob.glob(os.path.join(self.anomaly_dir, "*.jpg")))
        
       
        self.resize_shape = [img_size[0], img_size[1]]
        self.all_samples = self.normal_images + self.anomaly_images

    def __len__(self):
        return len(self.all_samples)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = cv2.GaussianBlur(image, (9,9), 0)
        
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
            
        image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
        mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[0]))
        
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1)) if mask.ndim == 3 else np.expand_dims(mask, 0)
        
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.all_samples[idx]
        is_anomaly = 1 if self.anomaly_dir in img_path else 0
        
        if is_anomaly:
            mask_name = os.path.basename(img_path).replace(".jpg", "_mask.png")
            mask_path = os.path.join(self.mask_dir, mask_name)
            image, mask = self.transform_image(img_path, mask_path)
        else:
            image, mask = self.transform_image(img_path, None)
            
        augmented_image = image.copy() 
        
        has_anomaly = np.array([is_anomaly], dtype=np.float32)
        
        sample = {
            'image': image,
            'anomaly_mask': mask,
            'augmented_image': augmented_image,
            'has_anomaly': has_anomaly,
            'idx': idx
        }
        return sample

class CustomTestDataset(Dataset):
    def __init__(self, data_path, classname, img_size):
        self.root_dir = os.path.join(data_path, 'test')
        self.images = sorted(glob.glob(os.path.join(self.root_dir, "*", "*.jpg")))
        self.resize_shape = [img_size[0], img_size[1]]
        self.classname = classname

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        # 保持与MVTecTestDataset完全一致的预处理流程
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = cv2.GaussianBlur(image, (9,9), 0)
        
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
            
        # 尺寸调整逻辑与MVTecTestDataset一致
        image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
        mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[0]))
        
        # 保持相同的归一化和维度处理
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, 0)  # 匹配MVTecTestDataset的维度 (1,H,W)

        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_name = file_name.replace(".jpg", "_mask.png")
            mask_path = os.path.join(dir_path, "../mask", mask_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {
            'image': image,
            'mask': mask,          
            'has_anomaly': has_anomaly,
            'idx': idx,
            'type': img_path[len(self.root_dir):-len(file_name)],
            'file_name': f"{base_dir}_{file_name}"
        }
        return sample