import numpy as np
import matplotlib.pyplot as plt
from hand_digital_dataloader import HandDigitalDataLoader
import importlib.util
import sys
import os

# 动态导入01cnn_basic_coponent.py模块
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '01cnn_basic_coponent.py')
spec = importlib.util.spec_from_file_location("cnn_basic_component", module_path)
cnn_basic_component = importlib.util.module_from_spec(spec)
sys.modules["cnn_basic_component"] = cnn_basic_component
spec.loader.exec_module(cnn_basic_component)

# 从模块中获取所需的函数（包括反向传播组件）
convolution2d = cnn_basic_component.convolution2d
convolution2d_multi_channel_batch = cnn_basic_component.convolution2d_multi_channel_batch
conv2dGradient = cnn_basic_component.conv2dGradient
conv2dGradient_batch = cnn_basic_component.conv2dGradient_batch

import warnings

# 过滤数值警告
warnings.filterwarnings('ignore')

# 设置随机种子，保证结果可复现
np.random.seed(42)

# ==================== 1. CNN网络组件定义 ====================

def relu(z):
    """ReLU激活函数"""
    return np.maximum(0, z)

def relu_derivative(a):
    """ReLU的导数"""
    return (a > 0).astype(float)

def softmax(z):
    """Softmax激活函数（输出层）- 数值稳定版本"""
    if z.ndim == 2:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def cross_entropy_loss(A, Y):
    """交叉熵损失函数（适用于softmax多分类）"""
    m = Y.shape[0]
    epsilon = 1e-8
    # 防止log(0)
    loss = -(1 / m) * np.sum(Y * np.log(A + epsilon))
    return loss

def one_hot_encode(y, num_classes):
    """将标签转换为one-hot编码"""
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

# ==================== 1.5 批量正则化（Batch Normalization） ====================

def batch_norm_forward(X, gamma, beta, epsilon=1e-8):
    """
    批量正则化前向传播
    
    参数:
        X: 输入数据，形状取决于输入类型
           - 如果是4D (batch, channels, height, width): 卷积层
           - 如果是2D (batch, features): 全连接层
        gamma: 缩放参数
        beta: 平移参数
        epsilon: 防止除以0的小常数
        
    返回:
        X_norm: 标准化后的输出
        cache: 用于反向传播的缓存
    """
    # 保存原始形状以便后续处理
    original_shape = X.shape
    
    if X.ndim == 4:
        # 卷积层: (batch, channels, height, width)
        # 在batch, height, width维度上计算均值和方差
        batch_size, channels, height, width = X.shape
        X_reshape = X.transpose(1, 0, 2, 3).reshape(channels, -1)
    else:
        # 全连接层: (batch, features)
        # 在batch维度上计算均值和方差
        X_reshape = X.T
    
    # 计算批次的均值和方差
    mean = np.mean(X_reshape, axis=1, keepdims=True)
    var = np.var(X_reshape, axis=1, keepdims=True)
    
    # 标准化
    X_norm_reshape = (X_reshape - mean) / np.sqrt(var + epsilon)
    
    # 恢复形状
    if X.ndim == 4:
        X_norm_reshape = X_norm_reshape.reshape(channels, batch_size, height, width)
        X_norm = X_norm_reshape.transpose(1, 0, 2, 3)
    else:
        X_norm = X_norm_reshape.T
    
    # 缩放和平移
    if X.ndim == 4:
        gamma_reshape = gamma.reshape(1, -1, 1, 1)
        beta_reshape = beta.reshape(1, -1, 1, 1)
    else:
        gamma_reshape = gamma.reshape(1, -1)
        beta_reshape = beta.reshape(1, -1)
    
    X_bn = gamma_reshape * X_norm + beta_reshape
    
    # 缓存用于反向传播
    cache = {
        'X_reshape': X_reshape,
        'X_norm_reshape': X_norm_reshape,
        'mean': mean,
        'var': var,
        'gamma': gamma,
        'beta': beta,
        'epsilon': epsilon,
        'original_shape': original_shape,
        'is_conv': X.ndim == 4
    }
    
    return X_bn, cache

def batch_norm_backward(dX_bn, cache):
    """
    批量正则化反向传播
    
    参数:
        dX_bn: 来自上层的梯度
        cache: 前向传播的缓存
        
    返回:
        dX: 相对于输入的梯度
        dgamma: 相对于gamma的梯度
        dbeta: 相对于beta的梯度
    """
    X_reshape = cache['X_reshape']
    X_norm_reshape = cache['X_norm_reshape']
    mean = cache['mean']
    var = cache['var']
    gamma = cache['gamma']
    epsilon = cache['epsilon']
    original_shape = cache['original_shape']
    is_conv = cache['is_conv']
    
    if is_conv:
        batch_size, channels, height, width = original_shape
        dX_bn_reshape = dX_bn.transpose(1, 0, 2, 3).reshape(channels, -1)
    else:
        dX_bn_reshape = dX_bn.T
    
    # 计算相对于beta的梯度
    if is_conv:
        dbeta = np.sum(dX_bn, axis=(0, 2, 3), keepdims=False)
    else:
        dbeta = np.sum(dX_bn, axis=0, keepdims=False)
    
    # 计算相对于gamma的梯度
    if is_conv:
        # 恢复X_norm的原始形状 (batch_size, channels, height, width)
        X_norm = X_norm_reshape.transpose(1, 0, 2, 3)
        dgamma = np.sum(dX_bn * X_norm, axis=(0, 2, 3), keepdims=False)
    else:
        dgamma = np.sum(dX_bn * X_norm_reshape.T, axis=0, keepdims=False)
    
    # 计算相对于标准化输入的梯度
    dX_norm_reshape = dX_bn_reshape * gamma.reshape(-1, 1)
    
    # 计算相对于方差的梯度
    dvar = np.sum(dX_norm_reshape * (X_reshape - mean) * -0.5 * (var + epsilon) ** (-1.5), axis=1, keepdims=True)
    
    # 计算相对于均值的梯度
    dmean = np.sum(dX_norm_reshape * -1 / np.sqrt(var + epsilon), axis=1, keepdims=True) + \
            dvar * np.sum(-2 * (X_reshape - mean), axis=1, keepdims=True) / X_reshape.shape[1]
    
    # 计算相对于输入的梯度
    dX_reshape = dX_norm_reshape / np.sqrt(var + epsilon) + \
                 dvar * 2 * (X_reshape - mean) / X_reshape.shape[1] + \
                 dmean / X_reshape.shape[1]
    
    # 恢复形状
    if is_conv:
        dX_reshape = dX_reshape.reshape(channels, batch_size, height, width)
        dX = dX_reshape.transpose(1, 0, 2, 3)
    else:
        dX = dX_reshape.T
    
    return dX, dgamma, dbeta

# ==================== 2. CNN前向传播 ====================

def cnn_forward(X_batch, params, use_bn=False):
    """
    CNN前向传播（无池化层）
    
    参数:
        X_batch: 输入批次数据 (batch_size, height, width)
        params: 网络参数字典
        
    返回:
        caches: 缓存各层的中间结果用于反向传播
        output: 最终输出 (batch_size, n_classes)
    """
    batch_size = X_batch.shape[0]
    X_batch = X_batch.reshape(batch_size, 1, X_batch.shape[1], X_batch.shape[2])

    # 第1层卷积: 100x100 -> 50x50 (使用3x3卷积核，stride=2, padding=0)
    W1 = params['W1'].reshape(params['W1'].shape[0], 1, params['W1'].shape[1], params['W1'].shape[2])
    b1 = params['b1']
    conv1_outputs = convolution2d_multi_channel_batch(X_batch, W1, padding=0, stride=2) + b1.reshape(1, -1, 1, 1)
    if use_bn:
        bn1, bn1_cache = batch_norm_forward(conv1_outputs, params['gamma1'], params['beta1'])
        a1 = relu(bn1)
    else:
        bn1_cache = None
        a1 = relu(conv1_outputs)

    # 第2层卷积: 50x50 -> 24x24 (使用3x3卷积核，stride=2, padding=0)
    W2 = params['W2']
    b2 = params['b2']
    conv2_outputs = convolution2d_multi_channel_batch(a1, W2, padding=0, stride=2) + b2.reshape(1, -1, 1, 1)
    if use_bn:
        bn2, bn2_cache = batch_norm_forward(conv2_outputs, params['gamma2'], params['beta2'])
        a2 = relu(bn2)
    else:
        bn2_cache = None
        a2 = relu(conv2_outputs)

    # 第3层卷积: 24x24 -> 11x11 (使用3x3卷积核，stride=2, padding=0)
    W3 = params['W3']
    b3 = params['b3']
    conv3_outputs = convolution2d_multi_channel_batch(a2, W3, padding=0, stride=2) + b3.reshape(1, -1, 1, 1)
    if use_bn:
        bn3, bn3_cache = batch_norm_forward(conv3_outputs, params['gamma3'], params['beta3'])
        a3 = relu(bn3)
    else:
        bn3_cache = None
        a3 = relu(conv3_outputs)

    # 第4层卷积: 11x11 -> 5x5 (使用3x3卷积核，stride=2, padding=0)
    W4 = params['W4']
    b4 = params['b4']
    conv4_outputs = convolution2d_multi_channel_batch(a3, W4, padding=0, stride=2) + b4.reshape(1, -1, 1, 1)
    if use_bn:
        bn4, bn4_cache = batch_norm_forward(conv4_outputs, params['gamma4'], params['beta4'])
        a4 = relu(bn4)
    else:
        bn4_cache = None
        a4 = relu(conv4_outputs)

    # 拉平: (batch_size, 256, output_h, output_w) -> (batch_size, flattened_size)
    flattened = a4.reshape(batch_size, -1)

    # 第1层全连接: flattened_size -> 512
    W5 = params['W5']
    b5 = params['b5']
    z5 = np.dot(flattened, W5) + b5.T
    if use_bn:
        bn5, bn5_cache = batch_norm_forward(z5, params['gamma5'], params['beta5'])
        a5 = relu(bn5)
    else:
        bn5_cache = None
        a5 = relu(z5)

    # 第2层全连接: 512 -> 10（输出层，不用BN）
    W6 = params['W6']
    b6 = params['b6']
    z6 = np.dot(a5, W6) + b6.T
    a6 = softmax(z6)

    # 缓存中间结果用于反向传播
    caches = {
        'X_batch': X_batch,
        'conv1_outputs': conv1_outputs,
        'bn1_cache': bn1_cache,
        'a1': a1,
        'conv2_outputs': conv2_outputs,
        'bn2_cache': bn2_cache,
        'a2': a2,
        'conv3_outputs': conv3_outputs,
        'bn3_cache': bn3_cache,
        'a3': a3,
        'conv4_outputs': conv4_outputs,
        'bn4_cache': bn4_cache,
        'a4': a4,
        'flattened': flattened,
        'z5': z5,
        'bn5_cache': bn5_cache,
        'a5': a5,
        'z6': z6,
        'a6': a6
    }

    return caches, a6

# ==================== 3. CNN反向传播 ====================

def cnn_backward(caches, Y, params, use_bn=False):
    """
    CNN反向传播（无池化层）
    
    参数:
        caches: 前向传播缓存
        Y: 真实标签 (batch_size, n_classes)
        params: 网络参数
        
    返回:
        grads: 各参数的梯度
    """
    batch_size = caches['X_batch'].shape[0]

    # 输出层梯度 (softmax + cross entropy)
    dz6 = caches['a6'] - Y  # (batch_size, 10)

    # 全连接层2梯度
    dW6 = np.dot(caches['a5'].T, dz6) / batch_size  # (512, 10)
    db6 = np.sum(dz6, axis=0, keepdims=True).T / batch_size  # (10, 1)

    # ReLU梯度
    da5 = np.dot(dz6, params['W6'].T)  # (batch_size, 512)
    dbn5 = da5 * relu_derivative(caches['a5'])  # (batch_size, 512)
    if use_bn and caches.get('bn5_cache') is not None:
        dz5, dgamma5, dbeta5 = batch_norm_backward(dbn5, caches['bn5_cache'])
    else:
        dz5 = dbn5
        dgamma5 = np.zeros_like(params.get('gamma5', np.array([])))
        dbeta5 = np.zeros_like(params.get('beta5', np.array([])))

    # 全连接层1梯度
    dflattened = np.dot(dz5, params['W5'].T)  # (batch_size, flattened_size)
    dW5 = np.dot(caches['flattened'].T, dz5) / batch_size  # (flattened_size, 512)
    db5 = np.sum(dz5, axis=0, keepdims=True).T / batch_size  # (512, 1)

    # 获取第4层卷积的输出通道数和空间尺寸
    num_output_channels_4 = params['W4'].shape[0]  # 256

    # 从缓存中获取第4层卷积输出的空间尺寸
    _, _, height_4, width_4 = caches['a4'].shape  # (batch_size, 256, 5, 5)

    # 重塑回卷积层输出形状
    da4 = dflattened.reshape(batch_size, num_output_channels_4, height_4, width_4)
    
    # 第4层ReLU梯度
    dbn4 = da4 * relu_derivative(caches['a4'])
    if use_bn and caches.get('bn4_cache') is not None:
        dconv4_outputs, dgamma4, dbeta4 = batch_norm_backward(dbn4, caches['bn4_cache'])
    else:
        dconv4_outputs = dbn4
        dgamma4 = np.zeros_like(params.get('gamma4', np.array([])))
        dbeta4 = np.zeros_like(params.get('beta4', np.array([])))

    # 第4层卷积梯度 - 批次级别矩阵化计算
    dx3, dW4, db4 = conv2dGradient_batch(
        dconv4_outputs,
        caches['a3'],
        params['W4'],
        stride=2,
        padding=0
    )

    # 第3层ReLU梯度
    dbn3 = dx3 * relu_derivative(caches['a3'])
    if use_bn and caches.get('bn3_cache') is not None:
        dconv3_outputs, dgamma3, dbeta3 = batch_norm_backward(dbn3, caches['bn3_cache'])
    else:
        dconv3_outputs = dbn3
        dgamma3 = np.zeros_like(params.get('gamma3', np.array([])))
        dbeta3 = np.zeros_like(params.get('beta3', np.array([])))

    # 第3层卷积梯度
    dx2, dW3, db3 = conv2dGradient_batch(
        dconv3_outputs,
        caches['a2'],
        params['W3'],
        stride=2,
        padding=0
    )

    # 第2层ReLU梯度
    dbn2 = dx2 * relu_derivative(caches['a2'])
    if use_bn and caches.get('bn2_cache') is not None:
        dconv2_outputs, dgamma2, dbeta2 = batch_norm_backward(dbn2, caches['bn2_cache'])
    else:
        dconv2_outputs = dbn2
        dgamma2 = np.zeros_like(params.get('gamma2', np.array([])))
        dbeta2 = np.zeros_like(params.get('beta2', np.array([])))

    # 第2层卷积梯度
    dx1, dW2, db2 = conv2dGradient_batch(
        dconv2_outputs,
        caches['a1'],
        params['W2'],
        stride=2,
        padding=0
    )

    # 第1层ReLU梯度
    dbn1 = dx1 * relu_derivative(caches['a1'])
    if use_bn and caches.get('bn1_cache') is not None:
        dconv1_outputs, dgamma1, dbeta1 = batch_norm_backward(dbn1, caches['bn1_cache'])
    else:
        dconv1_outputs = dbn1
        dgamma1 = np.zeros_like(params.get('gamma1', np.array([])))
        dbeta1 = np.zeros_like(params.get('beta1', np.array([])))

    # 第1层卷积梯度 - 注意 params['W1'] 存储为 (num_kernels, kH, kW)，需要 reshape
    W1_reshaped = params['W1'].reshape(params['W1'].shape[0], 1, params['W1'].shape[1], params['W1'].shape[2])
    _, dW1, db1 = conv2dGradient_batch(
        dconv1_outputs,
        caches['X_batch'],
        W1_reshaped,
        stride=2,
        padding=0
    )

    grads = {
        'dW6': dW6,
        'db6': db6,
        'dW5': dW5,
        'db5': db5,
        'dgamma5': dgamma5,
        'dbeta5': dbeta5,
        'dW4': dW4,
        'db4': db4,
        'dgamma4': dgamma4,
        'dbeta4': dbeta4,
        'dW3': dW3,
        'db3': db3,
        'dgamma3': dgamma3,
        'dbeta3': dbeta3,
        'dW2': dW2,
        'db2': db2,
        'dgamma2': dgamma2,
        'dbeta2': dbeta2,
        'dW1': dW1,
        'db1': db1,
        'dgamma1': dgamma1,
        'dbeta1': dbeta1
    }

    return grads

# ==================== 4. 参数初始化 ====================

def calculate_output_size(input_size, kernel_size, stride, padding=0):
    """计算卷积输出尺寸"""
    return (input_size - kernel_size + 2 * padding) // stride + 1

def initialize_parameters(input_height=100, input_width=100):
    """初始化CNN网络参数（带有批量正则化）"""
    params = {}
    
    # 卷积层参数
    params['W1'] = np.random.randn(32, 3, 3) * 0.1  # 32个3x3卷积核
    params['b1'] = np.zeros((32,))
    params['gamma1'] = np.ones((32,))  # BN参数
    params['beta1'] = np.zeros((32,))
    
    params['W2'] = np.random.randn(64, 32, 3, 3) * 0.1  # 64个3x3卷积核
    params['b2'] = np.zeros((64,))
    params['gamma2'] = np.ones((64,))  # BN参数
    params['beta2'] = np.zeros((64,))
    
    params['W3'] = np.random.randn(128, 64, 3, 3) * 0.1  # 128个3x3卷积核
    params['b3'] = np.zeros((128,))
    params['gamma3'] = np.ones((128,))  # BN参数
    params['beta3'] = np.zeros((128,))
    
    params['W4'] = np.random.randn(256, 128, 3, 3) * 0.1  # 256个3x3卷积核
    params['b4'] = np.zeros((256,))
    params['gamma4'] = np.ones((256,))  # BN参数
    params['beta4'] = np.zeros((256,))
    
    # 动态计算全连接层输入维度
    # 第1层: 100x100 -> 50x50 (stride=2, kernel=3)
    h1 = calculate_output_size(input_height, 3, 2)
    w1 = calculate_output_size(input_width, 3, 2)
    
    # 第2层: 50x50 -> 24x24 (stride=2, kernel=3)
    h2 = calculate_output_size(h1, 3, 2)
    w2 = calculate_output_size(w1, 3, 2)
    
    # 第3层: 24x24 -> 11x11 (stride=2, kernel=3)
    h3 = calculate_output_size(h2, 3, 2)
    w3 = calculate_output_size(w2, 3, 2)
    
    # 第4层: 11x11 -> 5x5 (stride=2, kernel=3)
    h4 = calculate_output_size(h3, 3, 2)
    w4 = calculate_output_size(w3, 3, 2)
    
    flattened_size = 256 * h4 * w4
    
    # 全连接层参数
    params['W5'] = np.random.randn(flattened_size, 512) * 0.1  # flattened_size -> 512
    params['b5'] = np.zeros((512, 1))
    params['gamma5'] = np.ones((512,))  # BN参数
    params['beta5'] = np.zeros((512,))
    
    params['W6'] = np.random.randn(512, 10) * 0.1  # 512 -> 10
    params['b6'] = np.zeros((10, 1))
    
    return params

# ==================== 5. 预测函数 ====================

def predict(X, params, use_bn=False):
    """预测函数"""
    # 重塑输入为(batch_size, height, width)
    if X.ndim == 2:
        batch_size = X.shape[1]
        X_reshaped = X.T.reshape(batch_size, 100, 100)
    else:
        X_reshaped = X

    _, output = cnn_forward(X_reshaped, params, use_bn=use_bn)
    predictions = np.argmax(output, axis=1)
    return predictions

def accuracy(Y_pre, Y_true):
    """准确率计算"""
    if Y_true.ndim > 1 and Y_true.shape[1] > 1:
        Y_true_flat = np.argmax(Y_true, axis=1)
    else:
        Y_true_flat = Y_true.flatten()

    Y_pre_flat = Y_pre.flatten()
    return np.mean(Y_pre_flat == Y_true_flat)

# ==================== 6. 训练函数 ====================

def train_cnn(loader, params, learning_rate=0.001, epochs=10, use_Adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8, use_bn=False):
    """训练CNN模型（无池化层）

    参数:
        use_Adam: 是否使用 Adam 优化器
        beta1, beta2, epsilon: Adam 超参数
    """
    loss_history = []
    train_acc_history = []
    test_acc_history = []
    
    # 获取测试集数据
    X_test_full, y_test_full = loader.get_test_data(one_hot=False)
    X_test_reshaped = X_test_full.T.reshape(-1, 100, 100)
    
    # 如果使用 Adam，初始化一阶、二阶矩估计
    if use_Adam:
        # conv layer 1 (params['W1'] stored as (out, KH, KW), gradients are (out, in, KH, KW))
        v_dW1 = np.zeros((params['W1'].shape[0], 1, params['W1'].shape[1], params['W1'].shape[2]))
        v_db1 = np.zeros_like(params['b1'])
        v_dgamma1 = np.zeros_like(params['gamma1'])
        v_dbeta1 = np.zeros_like(params['beta1'])
        s_dW1 = np.zeros((params['W1'].shape[0], 1, params['W1'].shape[1], params['W1'].shape[2]))
        s_db1 = np.zeros_like(params['b1'])
        s_dgamma1 = np.zeros_like(params['gamma1'])
        s_dbeta1 = np.zeros_like(params['beta1'])

        # conv layer 2
        v_dW2 = np.zeros_like(params['W2'])
        v_db2 = np.zeros_like(params['b2'])
        v_dgamma2 = np.zeros_like(params['gamma2'])
        v_dbeta2 = np.zeros_like(params['beta2'])
        s_dW2 = np.zeros_like(params['W2'])
        s_db2 = np.zeros_like(params['b2'])
        s_dgamma2 = np.zeros_like(params['gamma2'])
        s_dbeta2 = np.zeros_like(params['beta2'])

        # conv layer 3
        v_dW3 = np.zeros_like(params['W3'])
        v_db3 = np.zeros_like(params['b3'])
        v_dgamma3 = np.zeros_like(params['gamma3'])
        v_dbeta3 = np.zeros_like(params['beta3'])
        s_dW3 = np.zeros_like(params['W3'])
        s_db3 = np.zeros_like(params['b3'])
        s_dgamma3 = np.zeros_like(params['gamma3'])
        s_dbeta3 = np.zeros_like(params['beta3'])

        # conv layer 4
        v_dW4 = np.zeros_like(params['W4'])
        v_db4 = np.zeros_like(params['b4'])
        v_dgamma4 = np.zeros_like(params['gamma4'])
        v_dbeta4 = np.zeros_like(params['beta4'])
        s_dW4 = np.zeros_like(params['W4'])
        s_db4 = np.zeros_like(params['b4'])
        s_dgamma4 = np.zeros_like(params['gamma4'])
        s_dbeta4 = np.zeros_like(params['beta4'])

        # FC layers
        v_dW5 = np.zeros_like(params['W5'])
        v_db5 = np.zeros_like(params['b5'])
        v_dgamma5 = np.zeros_like(params['gamma5'])
        v_dbeta5 = np.zeros_like(params['beta5'])
        s_dW5 = np.zeros_like(params['W5'])
        s_db5 = np.zeros_like(params['b5'])
        s_dgamma5 = np.zeros_like(params['gamma5'])
        s_dbeta5 = np.zeros_like(params['beta5'])

        v_dW6 = np.zeros_like(params['W6'])
        v_db6 = np.zeros_like(params['b6'])
        s_dW6 = np.zeros_like(params['W6'])
        s_db6 = np.zeros_like(params['b6'])

    t = 0  # Adam 时间步计数
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        # 使用数据加载器获取所有批次
        for X_batch_flat, y_batch, batch_idx, total_batches in loader.get_all_batches():
            # 重塑输入数据为(batch_size, 100, 100)
            batch_size = X_batch_flat.shape[0]
            X_batch = X_batch_flat.reshape(batch_size, 100, 100)
            
            # 转换标签为one-hot编码
            Y_batch = one_hot_encode(y_batch, num_classes=10)
            
            # 前向传播
            caches, output = cnn_forward(X_batch, params, use_bn=use_bn)
            
            # 计算损失
            loss = cross_entropy_loss(output, Y_batch)
            epoch_loss += loss
            num_batches += 1
            
            # 反向传播（简化版）
            grads = cnn_backward(caches, Y_batch, params, use_bn=use_bn)
            
            # 更新参数（只更新全连接层和最后一层卷积层作为示例）
            if use_Adam:
                t += 1

                # 更新一阶矩
                v_dW6 = beta1 * v_dW6 + (1 - beta1) * grads['dW6']
                v_db6 = beta1 * v_db6 + (1 - beta1) * grads['db6']
                v_dW5 = beta1 * v_dW5 + (1 - beta1) * grads['dW5']
                v_db5 = beta1 * v_db5 + (1 - beta1) * grads['db5']
                if use_bn:
                    v_dgamma5 = beta1 * v_dgamma5 + (1 - beta1) * grads['dgamma5']
                    v_dbeta5 = beta1 * v_dbeta5 + (1 - beta1) * grads['dbeta5']
                
                v_dW4 = beta1 * v_dW4 + (1 - beta1) * grads['dW4']
                v_db4 = beta1 * v_db4 + (1 - beta1) * grads['db4']
                if use_bn:
                    v_dgamma4 = beta1 * v_dgamma4 + (1 - beta1) * grads['dgamma4']
                    v_dbeta4 = beta1 * v_dbeta4 + (1 - beta1) * grads['dbeta4']
                
                v_dW3 = beta1 * v_dW3 + (1 - beta1) * grads['dW3']
                v_db3 = beta1 * v_db3 + (1 - beta1) * grads['db3']
                if use_bn:
                    v_dgamma3 = beta1 * v_dgamma3 + (1 - beta1) * grads['dgamma3']
                    v_dbeta3 = beta1 * v_dbeta3 + (1 - beta1) * grads['dbeta3']
                
                v_dW2 = beta1 * v_dW2 + (1 - beta1) * grads['dW2']
                v_db2 = beta1 * v_db2 + (1 - beta1) * grads['db2']
                if use_bn:
                    v_dgamma2 = beta1 * v_dgamma2 + (1 - beta1) * grads['dgamma2']
                    v_dbeta2 = beta1 * v_dbeta2 + (1 - beta1) * grads['dbeta2']
                
                v_dW1 = beta1 * v_dW1 + (1 - beta1) * grads['dW1']
                v_db1 = beta1 * v_db1 + (1 - beta1) * grads['db1']
                if use_bn:
                    v_dgamma1 = beta1 * v_dgamma1 + (1 - beta1) * grads['dgamma1']
                    v_dbeta1 = beta1 * v_dbeta1 + (1 - beta1) * grads['dbeta1']

                # 更新二阶矩
                s_dW6 = beta2 * s_dW6 + (1 - beta2) * (grads['dW6'] ** 2)
                s_db6 = beta2 * s_db6 + (1 - beta2) * (grads['db6'] ** 2)
                s_dW5 = beta2 * s_dW5 + (1 - beta2) * (grads['dW5'] ** 2)
                s_db5 = beta2 * s_db5 + (1 - beta2) * (grads['db5'] ** 2)
                if use_bn:
                    s_dgamma5 = beta2 * s_dgamma5 + (1 - beta2) * (grads['dgamma5'] ** 2)
                    s_dbeta5 = beta2 * s_dbeta5 + (1 - beta2) * (grads['dbeta5'] ** 2)
                
                s_dW4 = beta2 * s_dW4 + (1 - beta2) * (grads['dW4'] ** 2)
                s_db4 = beta2 * s_db4 + (1 - beta2) * (grads['db4'] ** 2)
                if use_bn:
                    s_dgamma4 = beta2 * s_dgamma4 + (1 - beta2) * (grads['dgamma4'] ** 2)
                    s_dbeta4 = beta2 * s_dbeta4 + (1 - beta2) * (grads['dbeta4'] ** 2)
                
                s_dW3 = beta2 * s_dW3 + (1 - beta2) * (grads['dW3'] ** 2)
                s_db3 = beta2 * s_db3 + (1 - beta2) * (grads['db3'] ** 2)
                if use_bn:
                    s_dgamma3 = beta2 * s_dgamma3 + (1 - beta2) * (grads['dgamma3'] ** 2)
                    s_dbeta3 = beta2 * s_dbeta3 + (1 - beta2) * (grads['dbeta3'] ** 2)
                
                s_dW2 = beta2 * s_dW2 + (1 - beta2) * (grads['dW2'] ** 2)
                s_db2 = beta2 * s_db2 + (1 - beta2) * (grads['db2'] ** 2)
                if use_bn:
                    s_dgamma2 = beta2 * s_dgamma2 + (1 - beta2) * (grads['dgamma2'] ** 2)
                    s_dbeta2 = beta2 * s_dbeta2 + (1 - beta2) * (grads['dbeta2'] ** 2)
                s_dW1 = beta2 * s_dW1 + (1 - beta2) * (grads['dW1'] ** 2)
                s_db1 = beta2 * s_db1 + (1 - beta2) * (grads['db1'] ** 2)
                if use_bn:
                    s_dgamma1 = beta2 * s_dgamma1 + (1 - beta2) * (grads['dgamma1'] ** 2)
                    s_dbeta1 = beta2 * s_dbeta1 + (1 - beta2) * (grads['dbeta1'] ** 2)

                # 偏差修正
                v_dW6_corr = v_dW6 / (1 - beta1 ** t)
                v_db6_corr = v_db6 / (1 - beta1 ** t)
                v_dW5_corr = v_dW5 / (1 - beta1 ** t)
                v_db5_corr = v_db5 / (1 - beta1 ** t)
                if use_bn:
                    v_dgamma5_corr = v_dgamma5 / (1 - beta1 ** t)
                    v_dbeta5_corr = v_dbeta5 / (1 - beta1 ** t)
                
                v_dW4_corr = v_dW4 / (1 - beta1 ** t)
                v_db4_corr = v_db4 / (1 - beta1 ** t)
                if use_bn:
                    v_dgamma4_corr = v_dgamma4 / (1 - beta1 ** t)
                    v_dbeta4_corr = v_dbeta4 / (1 - beta1 ** t)

                v_dW3_corr = v_dW3 / (1 - beta1 ** t)
                v_db3_corr = v_db3 / (1 - beta1 ** t)
                if use_bn:
                    v_dgamma3_corr = v_dgamma3 / (1 - beta1 ** t)
                    v_dbeta3_corr = v_dbeta3 / (1 - beta1 ** t)
                
                v_dW2_corr = v_dW2 / (1 - beta1 ** t)
                v_db2_corr = v_db2 / (1 - beta1 ** t)
                if use_bn:
                    v_dgamma2_corr = v_dgamma2 / (1 - beta1 ** t)
                    v_dbeta2_corr = v_dbeta2 / (1 - beta1 ** t)
                
                v_dW1_corr = v_dW1 / (1 - beta1 ** t)
                v_db1_corr = v_db1 / (1 - beta1 ** t)
                if use_bn:
                    v_dgamma1_corr = v_dgamma1 / (1 - beta1 ** t)
                    v_dbeta1_corr = v_dbeta1 / (1 - beta1 ** t)

                s_dW6_corr = s_dW6 / (1 - beta2 ** t)
                s_db6_corr = s_db6 / (1 - beta2 ** t)
                s_dW5_corr = s_dW5 / (1 - beta2 ** t)
                s_db5_corr = s_db5 / (1 - beta2 ** t)
                if use_bn:
                    s_dgamma5_corr = s_dgamma5 / (1 - beta2 ** t)
                    s_dbeta5_corr = s_dbeta5 / (1 - beta2 ** t)
                
                s_dW4_corr = s_dW4 / (1 - beta2 ** t)
                s_db4_corr = s_db4 / (1 - beta2 ** t)
                if use_bn:
                    s_dgamma4_corr = s_dgamma4 / (1 - beta2 ** t)
                    s_dbeta4_corr = s_dbeta4 / (1 - beta2 ** t)

                s_dW3_corr = s_dW3 / (1 - beta2 ** t)
                s_db3_corr = s_db3 / (1 - beta2 ** t)
                if use_bn:
                    s_dgamma3_corr = s_dgamma3 / (1 - beta2 ** t)
                    s_dbeta3_corr = s_dbeta3 / (1 - beta2 ** t)
                
                s_dW2_corr = s_dW2 / (1 - beta2 ** t)
                s_db2_corr = s_db2 / (1 - beta2 ** t)
                if use_bn:
                    s_dgamma2_corr = s_dgamma2 / (1 - beta2 ** t)
                    s_dbeta2_corr = s_dbeta2 / (1 - beta2 ** t)
                
                s_dW1_corr = s_dW1 / (1 - beta2 ** t)
                s_db1_corr = s_db1 / (1 - beta2 ** t)
                if use_bn:
                    s_dgamma1_corr = s_dgamma1 / (1 - beta2 ** t)
                    s_dbeta1_corr = s_dbeta1 / (1 - beta2 ** t)

                # 参数更新
                params['W6'] = params['W6'] - (learning_rate / (np.sqrt(s_dW6_corr) + epsilon)) * v_dW6_corr
                params['b6'] = params['b6'] - (learning_rate / (np.sqrt(s_db6_corr) + epsilon)) * v_db6_corr
                params['W5'] = params['W5'] - (learning_rate / (np.sqrt(s_dW5_corr) + epsilon)) * v_dW5_corr
                params['b5'] = params['b5'] - (learning_rate / (np.sqrt(s_db5_corr) + epsilon)) * v_db5_corr
                if use_bn:
                    params['gamma5'] = params['gamma5'] - (learning_rate / (np.sqrt(s_dgamma5_corr) + epsilon)) * v_dgamma5_corr
                    params['beta5'] = params['beta5'] - (learning_rate / (np.sqrt(s_dbeta5_corr) + epsilon)) * v_dbeta5_corr
                
                params['W4'] = params['W4'] - (learning_rate / (np.sqrt(s_dW4_corr) + epsilon)) * v_dW4_corr
                params['b4'] = params['b4'] - (learning_rate / (np.sqrt(s_db4_corr) + epsilon)) * v_db4_corr
                if use_bn:
                    params['gamma4'] = params['gamma4'] - (learning_rate / (np.sqrt(s_dgamma4_corr) + epsilon)) * v_dgamma4_corr
                    params['beta4'] = params['beta4'] - (learning_rate / (np.sqrt(s_dbeta4_corr) + epsilon)) * v_dbeta4_corr
                
                params['W3'] = params['W3'] - (learning_rate / (np.sqrt(s_dW3_corr) + epsilon)) * v_dW3_corr
                params['b3'] = params['b3'] - (learning_rate / (np.sqrt(s_db3_corr) + epsilon)) * v_db3_corr
                if use_bn:
                    params['gamma3'] = params['gamma3'] - (learning_rate / (np.sqrt(s_dgamma3_corr) + epsilon)) * v_dgamma3_corr
                    params['beta3'] = params['beta3'] - (learning_rate / (np.sqrt(s_dbeta3_corr) + epsilon)) * v_dbeta3_corr
                
                params['W2'] = params['W2'] - (learning_rate / (np.sqrt(s_dW2_corr) + epsilon)) * v_dW2_corr
                params['b2'] = params['b2'] - (learning_rate / (np.sqrt(s_db2_corr) + epsilon)) * v_db2_corr
                if use_bn:
                    params['gamma2'] = params['gamma2'] - (learning_rate / (np.sqrt(s_dgamma2_corr) + epsilon)) * v_dgamma2_corr
                    params['beta2'] = params['beta2'] - (learning_rate / (np.sqrt(s_dbeta2_corr) + epsilon)) * v_dbeta2_corr

                # W1 存储为 (num_kernels, kH, kW)，但 v_dW1_corr 是 (out, in, KH, KW)，需要 squeeze 中间的 in 维
                v_dW1_corr_squeezed = v_dW1_corr[:, 0, :, :]
                s_dW1_corr_squeezed = s_dW1_corr[:, 0, :, :]
                params['W1'] = params['W1'] - (learning_rate / (np.sqrt(s_dW1_corr_squeezed) + epsilon)) * v_dW1_corr_squeezed
                params['b1'] = params['b1'] - (learning_rate / (np.sqrt(s_db1_corr) + epsilon)) * v_db1_corr
                if use_bn:
                    params['gamma1'] = params['gamma1'] - (learning_rate / (np.sqrt(s_dgamma1_corr) + epsilon)) * v_dgamma1_corr
                    params['beta1'] = params['beta1'] - (learning_rate / (np.sqrt(s_dbeta1_corr) + epsilon)) * v_dbeta1_corr
            else:
                params['W6'] -= learning_rate * grads['dW6']
                params['b6'] -= learning_rate * grads['db6']
                params['W5'] -= learning_rate * grads['dW5']
                params['b5'] -= learning_rate * grads['db5']
                if use_bn:
                    params['gamma5'] -= learning_rate * grads['dgamma5']
                    params['beta5'] -= learning_rate * grads['dbeta5']
                
                params['W4'] -= learning_rate * grads['dW4']
                params['b4'] -= learning_rate * grads['db4']
                if use_bn:
                    params['gamma4'] -= learning_rate * grads['dgamma4']
                    params['beta4'] -= learning_rate * grads['dbeta4']
                
                params['W3'] -= learning_rate * grads['dW3']
                params['b3'] -= learning_rate * grads['db3']
                if use_bn:
                    params['gamma3'] -= learning_rate * grads['dgamma3']
                    params['beta3'] -= learning_rate * grads['dbeta3']
                
                params['W2'] -= learning_rate * grads['dW2']
                params['b2'] -= learning_rate * grads['db2']
                if use_bn:
                    params['gamma2'] -= learning_rate * grads['dgamma2']
                    params['beta2'] -= learning_rate * grads['dbeta2']
                
                # grads['dW1'] 可能为 (out, in, KH, KW)，需要 squeeze 中间的 in 维
                if grads['dW1'].ndim == 4 and grads['dW1'].shape[1] == 1:
                    grads_dW1_squeezed = grads['dW1'][:, 0, :, :]
                else:
                    grads_dW1_squeezed = grads['dW1']
                params['W1'] -= learning_rate * grads_dW1_squeezed
                params['b1'] -= learning_rate * grads['db1']
                if use_bn:
                    params['gamma1'] -= learning_rate * grads['dgamma1']
                    params['beta1'] -= learning_rate * grads['dbeta1']
            
            if batch_idx % 1 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{total_batches} - Loss: {loss:.4f}")
        
        # 记录平均损失
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # 评估训练和测试准确率
        # 训练集评估（取一个小样本）
        train_sample_size = min(1000, loader.X_train.shape[0])
        train_indices = np.random.choice(loader.X_train.shape[0], train_sample_size, replace=False)
        X_train_sample = loader.X_train[train_indices].reshape(train_sample_size, 100, 100)
        y_train_sample = loader.y_train[train_indices]
        train_pred = predict(X_train_sample, params)
        train_acc = accuracy(train_pred, y_train_sample)
        train_acc_history.append(train_acc)
        
        # 测试集评估
        test_pred = predict(X_test_reshaped, params)
        test_acc = accuracy(test_pred, y_test_full)
        test_acc_history.append(test_acc)
        
        print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
    
    return loss_history, train_acc_history, test_acc_history, params

# ==================== 7. 主函数 ====================

def main():
    """主函数：训练并评估CNN模型（无池化层）"""
    
    print("="*60)
    print("手势数字 CNN 分类器（无池化层）")
    print("="*60)
    
    # 创建数据加载器，指定image_size=(100, 100)
    print("\n正在加载手势数字数据集...")
    loader = HandDigitalDataLoader(
        batch_size=32,  # 使用较小的batch size以适应内存限制
        shuffle=True,
        normalize=True,
        image_size=(100, 100),  # 指定输入尺寸为100x100
        grayscale=True
    )
    
    # 获取训练集信息
    info = loader.get_train_info()
    print(f"\n数据集信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 初始化网络参数
    print("\n初始化CNN网络参数...")
    params = initialize_parameters()
    
    # 训练模型
    print("\n开始训练CNN模型...")
    loss_history, train_acc_history, test_acc_history, trained_params = \
        train_cnn(loader, params, learning_rate=0.01, epochs=100, use_Adam=True)

    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(test_acc_history, label='Test Acc')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()