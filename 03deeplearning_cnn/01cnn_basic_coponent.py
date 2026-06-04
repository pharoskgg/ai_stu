import numpy as np
import matplotlib.pyplot as plt

def convolution2d(input, kernel, padding=0, stride=1):
    # 获取输入和卷积核的尺寸
    input_height, input_width = input.shape
    kernel_height, kernel_width = kernel.shape

    # 计算输出尺寸
    output_height = (input_height - kernel_height + 2 * padding) // stride + 1
    output_width = (input_width - kernel_width + 2 * padding) // stride + 1

    # 初始化输出矩阵
    output = np.zeros((output_height, output_width))

    # 对输入进行填充
    if padding > 0:
        input_padded = np.zeros((input_height + 2 * padding, input_width + 2 * padding))
        input_padded[padding:padding + input_height, padding:padding + input_width] = input
    else:
        input_padded = input
    
    # 进行卷积操作
    for i in range(0, output_height):
        for j in range(0, output_width):
            # 计算卷积核覆盖的输入区域
            input_region = input_padded[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
            
            # 计算卷积结果
            output[i, j] = np.sum(input_region * kernel)
    
    return output

def convolutionNd(input, kernel, padding=0, stride=1):
    # 多维卷积计算
    input_shape = input.shape
    kernel_shape = kernel.shape
    
    # 检查维度是否匹配
    if len(input_shape) != len(kernel_shape):
        raise ValueError(f"Input and kernel must have the same number of dimensions. Got {len(input_shape)} and {len(kernel_shape)}")
    
    # 计算输出尺寸
    output_shape = [(input_shape[i] - kernel_shape[i] + 2 * padding) // stride + 1 for i in range(len(input_shape))]
    output = np.zeros(output_shape)
    
    # 对输入进行填充
    if padding > 0:
        input_padded_shape = [input_shape[i] + 2 * padding for i in range(len(input_shape))]
        input_padded = np.zeros(input_padded_shape)
        slices = tuple(slice(padding, padding + input_shape[i]) for i in range(len(input_shape)))
        input_padded[slices] = input
    else:
        input_padded = input
    
    # 生成所有可能的输出位置索引
    indices = np.ndindex(*output_shape)
    
    # 对每个输出位置进行卷积计算
    for idx in indices:
        # 计算对应的输入区域切片
        # 这段切片
        slices = tuple(slice(idx[i] * stride, idx[i] * stride + kernel_shape[i]) for i in range(len(input_shape)))
        
        # 提取输入区域并计算卷积
        input_region = input_padded[slices]
        output[idx] = np.sum(input_region * kernel)
    
    return output

def pooling2d_max(input, pool_size=2, stride=2):
    # 获取输入尺寸
    input_height, input_width = input.shape

    # 计算输出尺寸
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1

    # 初始化输出矩阵
    output = np.zeros((output_height, output_width))

    # 进行最大池化操作
    for i in range(0, output_height):
        for j in range(0, output_width):
            # 计算池化区域
            input_region = input[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            # 取池化区域的最大值
            output[i, j] = np.max(input_region)

    return output

def pooling2d_avg(input, pool_size=2, stride=2):
    # 获取输入尺寸
    input_height, input_width = input.shape

    # 计算输出尺寸
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1

    # 创建输出矩阵
    output = np.zeros((output_height, output_width))

    # 进行平均池化操作
    for i in range(0, output_height):
        for j in range(0, output_width):
            # 计算池化区域
            input_region = input[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            # 取池化区域的平均值
            output[i, j] = np.mean(input_region)

    return output

def test_convolution2d():
    input = np.array([
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
    ])
    
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])
    
    output = convolution2d(input, kernel, padding=0, stride=1)
    print("Input:\n", input)
    print("Kernel:\n", kernel)
    print("Output:\n", output)

def test_convolutionNd():
    print("\n=== Testing 3D Convolution ===")
    
    # 创建一个3D输入 (4x4x4)
    input_3d = np.ones((4, 4, 4)) * 10
    
    # 创建一个3D卷积核 (2x2x2)
    kernel_3d = np.array([[[1, -1], [1, -1]], 
                          [[1, -1], [1, -1]]])
    
    output_3d = convolutionNd(input_3d, kernel_3d, padding=0, stride=1)
    print("3D Input shape:", input_3d.shape)
    print("3D Kernel shape:", kernel_3d.shape)
    print("3D Output shape:", output_3d.shape)
    print("3D Output:\n", output_3d)
    
    print("\n=== Testing 2D Convolution with convolutionNd ===")
    
    # 用convolutionNd测试2D情况，应该与convolution2d结果一致
    input_2d = np.array([
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
    ])
    
    kernel_2d = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])
    
    output_nd = convolutionNd(input_2d, kernel_2d, padding=0, stride=1)
    output_2d = convolution2d(input_2d, kernel_2d, padding=0, stride=1)
    
    print("convolutionNd result matches convolution2d:", np.allclose(output_nd, output_2d))
    print("Output:\n", output_nd)

if __name__ == "__main__":
    # test_convolution2d()
    test_convolutionNd()
