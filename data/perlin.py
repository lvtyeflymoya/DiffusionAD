import torch
import math
import numpy as np

def lerp_np(x,y,w):
    # 计算x和y之间的线性插值，w为权重
    fin_out = (y-x)*w + x
    return fin_out

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    # 生成指定形状和分辨率的分形噪声
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        # 通过叠加不同频率和振幅的Perlin噪声生成分形噪声
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
 
    # 计算网格的分辨率增量
    delta = (res[0] / shape[0], res[1] / shape[1])
    # 计算每个小格子在大网格中的重复次数
    d = (shape[0] // res[0], shape[1] // res[1])
    # 生成基础网格
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    # 生成随机角度
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    # 计算角度对应的梯度向量
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    # 重复梯度向量以匹配目标形状
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    # 计算每个点的噪声值
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    # 计算平滑函数值
    t = f(grid)
    # 在水平方向进行线性插值
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    # 在垂直方向进行线性插值并返回结果
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    # 生成指定形状和分辨率的2D Perlin噪声
    delta = (res[0] / shape[0], res[1] / shape[1]) # 计算每个单元格的间隔
    d = (shape[0] // res[0], shape[1] // res[1])  # 计算每个分辨率单元格需要重复的次数
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1   # 生成网格坐标，并在每个维度上取模

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)    # 生成随机角度
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)  # 根据角度生成梯度向量
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1) # 扩展梯度以匹配所需形状的分辨率

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0]) # 计算左下角点的噪声值
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0]) # 计算右下角点的噪声值
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1]) # 计算左上角点的噪声值
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1]) # 计算右上角点的噪声值
    t = fade(grid[:shape[0], :shape[1]]) # 应用平滑函数
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]) # 线性插值计算最终的噪声值

"""
def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    #grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1]), indexing='ij'), dim=-1) % 1
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(
        0, res[1], delta[1])), dim=-1) % 1    
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
                torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
"""

