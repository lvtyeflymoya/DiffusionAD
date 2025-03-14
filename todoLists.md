1.获取异常合成后的图像
    已知：MVTectrainDataset的__getitem__会返回sample（字典），包含合成后的图像
    写一个测试，构造MVTectrainDataset类的对象，调用__getitem__方法，并保存图像（完成）
2.确定异常图像该怎么合成
    2.1现有合成过程：
    ![alt text](image.png)
    2.2新的合成方法：

3.模型参数量和计算量（MACs）
（重建网络）UNetModel计算量 (MACs): 278.548250624
（重建网络）UNetModel参数量: 131.578627
SegmentationSubNetwork计算量 (MACs): 38.239076352
SegmentationSubNetwork参数量: 28.373569
总参数量: 159.952196 M
总计算量 (MACs): 316.787326976 G

3060gpu的Tensor TFLOPS= 101  = 101000GFLOPS
FP32 TFLOPs = 12.7
FP64 TFLOPs = 6.3
A100gpu的Tensor TFLOPS = 156
FP32 TFLOPs = 19.5
FP64 TFLOPs = 9.7
因此粗略计算3060满功耗时一秒可以处理的图像数量为：101000/316=319张（Tensor TFLOPS）
                                            12.7 * 1000/316=40张（FP32 TFLOPs）
                                            6.3 * 1000/316=20张（FP64 TFLOPs）
因此粗略计算A100满功耗时一秒可以处理的图像数量为：156000/316=593张（Tensor TFLOPS）
                                            12.7 * 1000/316=61张（FP32 TFLOPs）
                                            6.3 * 1000/316=30张（FP64 TFLOPs）

4、楼上电脑启动tensorboard的命令，因为python.exe路径变了，但我还没查清拷贝过来的程序在为什么会调我笔记本上的python.exe路径
&E:\zhanglelelelelelele\virtual_envs\anomalib_env/python.exe "E:\zhanglelelelelelele\virtual_envs\anomalib_env\Scripts\tensorboard.exe" --logdir=E:\zhanglelelelelelele\Python_project\DiffusionAD_shipwood_ano_src\logs


4.实验分析

实验1: 布料类别 + DTD异常源
- epoch: 
- Loss
  - 重建损失
  - 分割损失
  - 总损失
- 指标
  - AU-ROC
  - Precision
  - Recall
- 存在问题: 

实验2: 非布料类别(纯自增强)
- epoch: 129
- Loss
  - 重建损失
  - 分割损失
  - 总损失
- 指标
  - AU-ROC
  - Precision
  - Recall
- 存在问题: 

实验3: 布料类别 + 自制异常源
- epoch: 30
- 训练时长
- Loss
  - 重建损失
  - 分割损失
  - 总损失
- 指标
  - AU-ROC
  - Precision
  - Recall
- 存在问题: 
- 
