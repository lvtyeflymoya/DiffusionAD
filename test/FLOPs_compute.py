import torch
from torchsummary import summary
from thop import profile
import os
import sys  
# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Recon_subnetwork import UNetModel
from models.Seg_subnetwork import SegmentationSubNetwork  
# 假设的参数
args = {
    'img_size': [256, 256],
    'base_channels': 128,
    'channel_mults': "",
    'dropout': 0,
    'num_heads': 4,
    'num_head_channels': -1,
    'channels': 3
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
unet_model = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args["dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"], in_channels=args["channels"]).to(device)
seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)



# # 计算参数量
# print("UNetModel参数量:")
# summary(unet_model, input_size=(args["channels"], args['img_size'][0], args['img_size'][1]))

print("\nSegmentationSubNetwork参数量:")
summary(seg_model, input_size=(6, args['img_size'][0], args['img_size'][1]))

# 计算计算量（FLOPs）
input_tensor = torch.randn(1, args["channels"], args['img_size'][0], args['img_size'][1]).to(device)
macs, params = profile(unet_model, inputs=(input_tensor, torch.tensor([1]).to(device)))
print(f"\nUNetModel计算量 (MACs): {macs / 1e9}")
print(f"UNetModel参数量: {params / 1e6}")

input_tensor_seg = torch.randn(1, 6, args['img_size'][0], args['img_size'][1]).to(device)
macs_seg, params_seg = profile(seg_model, inputs=(input_tensor_seg,))
print(f"\nSegmentationSubNetwork计算量 (MACs): {macs_seg / 1e9}")
print(f"SegmentationSubNetwork参数量: {params_seg / 1e6}")

# 计算总参数量和计算量
total_params = params + params_seg
total_macs = macs + macs_seg
print(f"\n总参数量: {total_params / 1e6} M")
print(f"总计算量 (MACs): {total_macs / 1e9} G")