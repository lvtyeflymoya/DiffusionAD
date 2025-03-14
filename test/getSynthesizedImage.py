import os
import cv2
import torch
import numpy as np
import sys  

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_beta_thresh import MVTecTrainDataset


data_dir = "E:/zhanglelelelelelele/ImageDataSet/DiffusionAD_dataset_shipwood_ano_src"
output_dir = "./outputs/augmented_images"
os.makedirs(output_dir, exist_ok=True)

# 初始化数据集
args = {
    "anomaly_source_path": "E:/zhanglelelelelelele/ImageDataSet/Diffusion_Dataset_test/carpet/test/wood_waterweed",  # 异常源的路径
    "mvtec_root_path": "E:/zhanglelelelelelele/ImageDataSet/DiffusionAD_dataset_shipwood_ano_src"   # 数据集根目录
}
dataset = MVTecTrainDataset(
    data_path=os.path.join(data_dir, "carpet"),  # 类别
    classname="carpet",
    img_size=[256, 256],
    args=args
)

# 保存20个增强样本
for i in range(50):
    sample = dataset[i]
    
    # 转换张量到numpy格式
    augmented = sample['augmented_image'].transpose(1, 2, 0) * 255
    original = sample['image'].transpose(1, 2, 0) * 255
    mask = sample['anomaly_mask'].transpose(1, 2, 0) * 255
    
    # 保存图像
    cv2.imwrite(os.path.join(output_dir, f"{i}_aug.jpg"), augmented[..., ::-1])  # RGB转BGR
    # cv2.imwrite(os.path.join(output_dir, f"{i}_ori.jpg"), original[..., ::-1])
    cv2.imwrite(os.path.join(output_dir, f"{i}_mask.jpg"), mask)

print(f"Saved augmented images to {output_dir}")