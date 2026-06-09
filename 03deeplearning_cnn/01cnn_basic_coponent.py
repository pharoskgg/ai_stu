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

def convolution3d(input, kernel, padding=0, stride=1):
    # 获取输入和卷积核的尺寸
    input_depth, input_height, input_width = input.shape
    kernel_depth, kernel_height, kernel_width = kernel.shape

    # 计算输出尺寸
    output_depth = (input_depth - kernel_depth + 2 * padding) // stride + 1
    output_height = (input_height - kernel_height + 2 * padding) // stride + 1
    output_width = (input_width - kernel_width + 2 * padding) // stride + 1

    # 创建输出矩阵 (保持与输入相同的维度顺序: depth, height, width)
    output = np.zeros((output_depth, output_height, output_width))

    # 对输入进行填充
    if padding > 0:
        input_padded = np.zeros((input_depth + 2 * padding, input_height + 2 * padding, input_width + 2 * padding))
        input_padded[padding:padding + input_depth, padding:padding + input_height, padding:padding + input_width] = input
    else:
        input_padded = input

    # 进行3D卷积操作
    for d in range(output_depth):
        for h in range(output_height):
            for w in range(output_width):
                # 计算卷积核覆盖的输入区域
                input_region = input_padded[d*stride:d*stride+kernel_depth, h*stride:h*stride+kernel_height, w*stride:w*stride+kernel_width]

                # 计算卷积结果
                output[d, h, w] = np.sum(input_region * kernel)

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

def pooling3d_max(input, pool_size=2, stride=2):
    # 获取输入尺寸
    input_depth, input_height, input_width = input.shape

    # 计算输出尺寸
    output_depth = (input_depth - pool_size) // stride + 1
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1

    # 创建输出矩阵 (保持与输入相同的维度顺序: depth, height, width)
    output = np.zeros((output_depth, output_height, output_width))

    # 进行3D最大池化操作
    for d in range(output_depth):
        for h in range(output_height):
            for w in range(output_width):
                # 计算池化区域
                input_region = input[d*stride:d*stride+pool_size, h*stride:h*stride+pool_size, w*stride:w*stride+pool_size]
                # 取池化区域的最大值
                output[d, h, w] = np.max(input_region)

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

def pooling3d_avg(input, pool_size=2, stride=2):
    # 获取输入尺寸
    input_depth, input_height, input_width = input.shape

    # 计算输出尺寸
    output_depth = (input_depth - pool_size) // stride + 1
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1

    # 创建输出矩阵 (保持与输入相同的维度顺序: depth, height, width)
    output = np.zeros((output_depth, output_height, output_width))

    # 进行3D平均池化操作
    for d in range(output_depth):
        for h in range(output_height):
            for w in range(output_width):
                # 计算池化区域
                input_region = input[d*stride:d*stride+pool_size, h*stride:h*stride+pool_size, w*stride:w*stride+pool_size]
                # 取池化区域的平均值
                output[d, h, w] = np.mean(input_region)

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

def test_convolution3d():
    print("\n=== Testing 3D Convolution ===")
    
    # 创建一个3D输入 (4x4x4)
    input_3d = np.ones((4, 4, 4)) * 10
    
    # 创建一个3D卷积核 (2x2x2)
    kernel_3d = np.array([[[1, -1], [1, -1]], 
                          [[1, -1], [1, -1]]])
    
    output_3d = convolution3d(input_3d, kernel_3d, padding=0, stride=1)
    print("3D Input shape:", input_3d.shape)
    print("3D Kernel shape:", kernel_3d.shape)
    print("3D Output shape:", output_3d.shape)
    print("3D Output:\n", output_3d)
    
    # 验证输出形状是否正确
    expected_shape = (3, 3, 3)  # (4-2+1, 4-2+1, 4-2+1)
    assert output_3d.shape == expected_shape, f"Expected shape {expected_shape}, got {output_3d.shape}"
    print("✓ 3D卷积输出形状正确")


def test_pooling2d_max():
    print("\n=== Testing 2D Max Pooling ===")

    input = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [3, 2, 4, 7],
        [8, 1, 3, 2]
    ])

    output = pooling2d_max(input, pool_size=2, stride=2)
    print("Input:\n", input)
    print("Output:\n", output)

    # 验证输出形状: (4-2)//2+1 = 2
    expected_shape = (2, 2)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print("✓ 2D Max Pooling 输出形状正确")

    # 验证输出值: 每个 2x2 区域取最大值
    expected = np.array([
        [6, 4],
        [8, 7]
    ])
    assert np.array_equal(output, expected), f"Expected\n{expected}\nGot\n{output}"
    print("✓ 2D Max Pooling 输出值正确")


def test_pooling3d_max():
    print("\n=== Testing 3D Max Pooling ===")

    # 创建一个3D输入 (4x4x4)
    input_3d = np.arange(64).reshape(4, 4, 4).astype(float)
    print("3D Input shape:", input_3d.shape)
    print("3D Input:\n", input_3d)

    output_3d = pooling3d_max(input_3d, pool_size=2, stride=2)
    print("3D Output shape:", output_3d.shape)
    print("3D Output:\n", output_3d)

    # 验证输出形状: (4-2)//2+1 = 2
    expected_shape = (2, 2, 2)
    assert output_3d.shape == expected_shape, f"Expected shape {expected_shape}, got {output_3d.shape}"
    print("✓ 3D Max Pooling 输出形状正确")

    # 验证第一个元素: input_3d[0:2, 0:2, 0:2] 的最大值
    expected_val = np.max(input_3d[0:2, 0:2, 0:2])
    assert output_3d[0, 0, 0] == expected_val, f"Expected {expected_val}, got {output_3d[0, 0, 0]}"
    print("✓ 3D Max Pooling 输出值正确")


def test_pooling3d_avg():
    print("\n=== Testing 3D Avg Pooling ===")

    # 创建一个3D输入 (4x4x4)
    input_3d = np.arange(64).reshape(4, 4, 4).astype(float)
    print("3D Input shape:", input_3d.shape)
    print("3D Input:\n", input_3d)

    output_3d = pooling3d_avg(input_3d, pool_size=2, stride=2)
    print("3D Output shape:", output_3d.shape)
    print("3D Output:\n", output_3d)

    # 验证输出形状: (4-2)//2+1 = 2
    expected_shape = (2, 2, 2)
    assert output_3d.shape == expected_shape, f"Expected shape {expected_shape}, got {output_3d.shape}"
    print("✓ 3D Avg Pooling 输出形状正确")

    # 验证第一个元素: input_3d[0:2, 0:2, 0:2] 的平均值
    expected_val = np.mean(input_3d[0:2, 0:2, 0:2])
    assert np.isclose(output_3d[0, 0, 0], expected_val), f"Expected {expected_val}, got {output_3d[0, 0, 0]}"
    print("✓ 3D Avg Pooling 输出值正确")


def test_pooling2d_avg():
    print("\n=== Testing 2D Avg Pooling ===")

    input = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [3, 2, 4, 7],
        [8, 1, 3, 2]
    ], dtype=float)

    output = pooling2d_avg(input, pool_size=2, stride=2)
    print("Input:\n", input)
    print("Output:\n", output)

    # 验证输出形状: (4-2)//2+1 = 2
    expected_shape = (2, 2)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print("✓ 2D Avg Pooling 输出形状正确")

    # 验证输出值: 每个 2x2 区域取平均值
    expected = np.array([
        [np.mean([1, 3, 5, 6]), np.mean([2, 4, 1, 2])],
        [np.mean([3, 2, 8, 1]), np.mean([4, 7, 3, 2])]
    ])
    assert np.allclose(output, expected), f"Expected\n{expected}\nGot\n{output}"
    print("✓ 2D Avg Pooling 输出值正确")


def test():
    test_pooling2d_max()
    test_pooling3d_max()
    test_pooling3d_avg()
    test_pooling2d_avg()

if __name__ == "__main__":
    test()
