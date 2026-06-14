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

def upsample(input, stride):
    """
    上采样：仅保留原始点，中间插值点全部填 0
    输出形状 = 原形状 * stride - (stride - 1)
    """
    # 计算输出形状
    out_shape = tuple(s * stride - (stride - 1) for s in input.shape)
    output = np.zeros(out_shape, dtype=input.dtype)

    # 生成原始点在输出中的坐标（每隔 stride 取一个位置）
    input_indices = [np.arange(s) * stride for s in input.shape]
    grid = np.meshgrid(*input_indices, indexing='ij')

    # 只把原始点填进去，其余都是 0
    output[tuple(grid)] = input

    return output
 
def conv2dGradient(outGradent, input, kernel, stride=1, padding=0):
    input_height, input_width = input.shape
    kernel_height, kernel_width = kernel.shape
    outGradent_height, outGradent_width = outGradent.shape

    if padding > kernel_height // 2 or padding > kernel_width // 2:
        raise ValueError("Padding cannot be greater than half of the kernel size.")

    input_gradient = np.zeros_like(input)
    kernel_gradient = np.zeros_like(kernel)

    input_pad = np.pad(input, ((padding, padding), (padding, padding)), mode='constant')

    kernel_gradient = convolution2d(outGradent, kernel, stride=1, padding=0)


    bias_gradient = np.sum(outGradent)

    outGradent_upsample = upsample(outGradent, stride)
    pad_h = kernel_height - 1 - padding
    pad_w = kernel_width - 1 - padding
    OutGradent_up_paded = np.pad(outGradent_upsample, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    kernel = np.flipud(np.fliplr(kernel)) # 卷积核整体旋转 180，左右，上下翻转
    input_gradient = convolution2d(OutGradent_up_paded, kernel, stride=1, padding=0)


    return input_gradient, kernel_gradient, bias_gradient

def gradient_pooling_max(input, outGradient, pool_size=2, stride=2):
    input_height, input_width = input.shape
    outGradient_height, outGradient_width = outGradient.shape

    input_gradient = np.zeros_like(input)

    for i in range(outGradient_height):
        for j in range(outGradient_width):
            h_start = i * stride
            w_start = j * stride
            h_end = min(h_start + pool_size, input_height)
            w_end = min(w_start + pool_size, input_width)

            input_region = input[h_start:h_end, w_start:w_end]
            max_value = np.max(input_region)

            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    if input[h, w] == max_value:
                        input_gradient[h, w] += outGradient[i, j]

    return input_gradient
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

def test_upsample():
    print("\n=== Testing Upsample ===")
    
    input = np.array([[1, 2], [3, 4]])
    stride = 2
    output = upsample(input, stride)
    
    print("Input:\n", input)
    print("Upsampled Output:\n", output)
    
    expected_output = np.array([[1, 0, 2],
                                [0, 0, 0],
                                [3, 0, 4]])
    
    assert np.array_equal(output, expected_output), f"Expected\n{expected_output}\nGot\n{output}"
    print("✓ Upsample 输出正确")

def test_Gradient():
    print("\n=== Testing Gradient ===")
    input = np.array([[1, 2, 3],
                      [0, 1, 4],
                      [1, 0, 2]], dtype=float)
    kernel = np.array([[1, 0],
                       [0, 1]], dtype=float)
    outGradent = np.array([[1, 2],
                          [3, 4]], dtype=float)
    stride = 1
    padding = 0
    input_gradient, kernel_gradient, bias_gradient = conv2dGradient(outGradent, input, kernel, stride, padding)
    print("Input Gradient:\n", input_gradient)
    print("Kernel Gradient:\n", kernel_gradient)
    print("Bias Gradient:\n", bias_gradient)
    assert np.array_equal(input_gradient, np.array([[1, 2, 0],
                                                   [3, 5, 2],
                                                   [0, 3, 4]])), "Input gradient 不正确"

def test():
    test_convolution3d()
    test_convolution2d()
    test_pooling2d_max()
    test_pooling3d_max()
    test_pooling3d_avg()
    test_pooling2d_avg()
    test_upsample()
    test_Gradient()

if __name__ == "__main__":
    test()
