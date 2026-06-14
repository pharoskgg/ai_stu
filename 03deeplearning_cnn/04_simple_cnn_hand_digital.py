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

# 从模块中获取所需的函数
convolution2d = cnn_basic_component.convolution2d
convolution2d_batch = cnn_basic_component.convolution2d_batch
conv2dGradient = cnn_basic_component.conv2dGradient

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
    # 减去最大值以防止数值溢出
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def cross_entropy_loss(A, Y):
    """交叉熵损失函数（适用于softmax多分类）"""
    m = Y.shape[1]
    epsilon = 1e-8
    # 防止log(0)
    loss = -(1 / m) * np.sum(Y * np.log(A + epsilon))
    return loss

def one_hot_encode(y, num_classes):
    """将标签转换为one-hot编码"""
    m = y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[y, np.arange(m)] = 1
    return one_hot

# ==================== 1.5 批量正则化 ====================

def batch_norm_forward(X, gamma, beta, epsilon=1e-8):
    """批量正则化前向传播，支持卷积层和全连接层"""
    cache = {
        'is_conv': X.ndim == 4,
        'epsilon': epsilon,
        'gamma': gamma,
        'beta': beta
    }
    if X.ndim == 4:
        batch_size, channels, height, width = X.shape
        X_flat = X.transpose(1, 0, 2, 3).reshape(channels, -1)
        mean = np.mean(X_flat, axis=1, keepdims=True)
        var = np.var(X_flat, axis=1, keepdims=True)
        X_norm_flat = (X_flat - mean) / np.sqrt(var + epsilon)
        X_norm = X_norm_flat.reshape(channels, batch_size, height, width).transpose(1, 0, 2, 3)
        out = gamma.reshape(1, -1, 1, 1) * X_norm + beta.reshape(1, -1, 1, 1)
        cache.update({'X_norm': X_norm, 'mean': mean, 'var': var, 'X_flat': X_flat})
    else:
        X_flat = X.T  # shape (features, batch)
        mean = np.mean(X_flat, axis=1, keepdims=True)
        var = np.var(X_flat, axis=1, keepdims=True)
        X_norm_flat = (X_flat - mean) / np.sqrt(var + epsilon)
        X_norm = X_norm_flat.T
        out = gamma.reshape(1, -1) * X_norm + beta.reshape(1, -1)
        cache.update({'X_norm': X_norm, 'mean': mean, 'var': var, 'X_flat': X_flat})
    return out, cache

def batch_norm_backward(dout, cache):
    """批量正则化反向传播"""
    is_conv = cache['is_conv']
    epsilon = cache['epsilon']
    gamma = cache['gamma']
    beta = cache['beta']
    X_norm = cache['X_norm']
    mean = cache['mean']
    var = cache['var']
    X_flat = cache['X_flat']

    if is_conv:
        batch_size, channels, height, width = dout.shape
        dgamma = np.sum(dout * X_norm, axis=(0, 2, 3))
        dbeta = np.sum(dout, axis=(0, 2, 3))
        dX_norm = dout * gamma.reshape(1, -1, 1, 1)
        dX_norm_flat = dX_norm.transpose(1, 0, 2, 3).reshape(channels, -1)
        m = dX_norm_flat.shape[1]
        dvar = np.sum(dX_norm_flat * (X_flat - mean) * -0.5 * (var + epsilon) ** (-1.5), axis=1, keepdims=True)
        dmean = np.sum(dX_norm_flat * -1.0 / np.sqrt(var + epsilon), axis=1, keepdims=True) + \
                dvar * np.sum(-2 * (X_flat - mean), axis=1, keepdims=True) / m
        dX_flat = dX_norm_flat / np.sqrt(var + epsilon) + dvar * 2 * (X_flat - mean) / m + dmean / m
        dX = dX_flat.reshape(channels, batch_size, height, width).transpose(1, 0, 2, 3)
    else:
        batch_size, features = dout.shape
        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        dX_norm = dout * gamma.reshape(1, -1)
        dX_flat = dX_norm.T  # (features, batch)
        m = dX_flat.shape[1]
        dvar = np.sum(dX_flat * (X_flat - mean) * -0.5 * (var + epsilon) ** (-1.5), axis=1, keepdims=True)
        dmean = np.sum(dX_flat * -1.0 / np.sqrt(var + epsilon), axis=1, keepdims=True) + \
                dvar * np.sum(-2 * (X_flat - mean), axis=1, keepdims=True) / m
        dX_flat = dX_flat / np.sqrt(var + epsilon) + dvar * 2 * (X_flat - mean) / m + dmean / m
        dX = dX_flat.T
    return dX, dgamma, dbeta

# ==================== 2. 简化CNN前向传播 ====================

def simple_cnn_forward(X_batch, params, use_bn=False):
    """
    简化CNN前向传播（1个卷积层 + 全连接层）
    
    参数:
        X_batch: 输入批次数据 (batch_size, height, width)
        params: 网络参数字典
        use_bn: 是否使用Batch Normalization
        
    返回:
        caches: 缓存各层的中间结果用于反向传播
        output: 最终输出 (n_classes, batch_size)
    """
    batch_size = X_batch.shape[0]
    
    # 第1层卷积: input_size -> output_size (使用3x3卷积核，stride=1, padding=0)
    W1 = params['W1']  # (2, 3, 3) - 2个3x3卷积核
    b1 = params['b1']  # (2,)

    conv1_outputs = convolution2d_batch(X_batch, W1, padding=0, stride=1)  # (batch_size, 2, output_h, output_w)
    conv1_outputs = conv1_outputs + b1.reshape(1, -1, 1, 1)
    if use_bn:
        conv1_outputs, bn1_cache = batch_norm_forward(conv1_outputs, params['gamma1'], params['beta1'])
    a1 = relu(conv1_outputs)

    # 拉平: (batch_size, 2, output_h, output_w) -> (batch_size, flattened_size)
    flattened = a1.reshape(batch_size, -1)  # (batch_size, flattened_size)

    # 隐藏层全连接: flattened_size -> hidden_size
    W2 = params['W2']  # (flattened_size, hidden_size)
    b2 = params['b2']  # (hidden_size, 1)
    z2 = np.dot(flattened, W2) + b2.T  # (batch_size, hidden_size)
    if use_bn:
        z2, bn2_cache = batch_norm_forward(z2, params['gamma2'], params['beta2'])
    a2 = relu(z2)  # (batch_size, hidden_size)

    # 输出层全连接: hidden_size -> 10
    W3 = params['W3']  # (hidden_size, 10)
    b3 = params['b3']  # (10, 1)
    z3 = np.dot(a2, W3) + b3.T  # (batch_size, 10)
    a3 = softmax(z3.T)  # (10, batch_size)

    # 缓存中间结果用于反向传播
    caches = {
        'X_batch': X_batch,
        'conv1_outputs': conv1_outputs,
        'a1': a1,
        'flattened': flattened,
        'z2': z2,
        'a2': a2,
        'z3': z3,
        'a3': a3,
        'use_bn': use_bn
    }
    if use_bn:
        caches['bn1_cache'] = bn1_cache
        caches['bn2_cache'] = bn2_cache

    return caches, a3

# ==================== 3. 简化CNN反向传播 ====================

def simple_cnn_backward(caches, Y, params, use_bn=False):
    """
    简化CNN反向传播（1个卷积层 + 全连接层）
    
    参数:
        caches: 前向传播缓存
        Y: 真实标签 (n_classes, batch_size)
        params: 网络参数
        
    返回:
        grads: 各参数的梯度
    """
    batch_size = caches['X_batch'].shape[0]

    # 输出层梯度 (softmax + cross entropy)
    dz3 = caches['a3'] - Y  # (10, batch_size)
    dz3_T = dz3.T  # (batch_size, 10)

    # 输出全连接层梯度
    dW3 = np.dot(caches['a2'].T, dz3_T) / batch_size  # (hidden_size, 10)
    db3 = np.sum(dz3, axis=1, keepdims=True) / batch_size  # (10, 1)

    # 将输出梯度反传到隐藏层
    da2 = np.dot(dz3_T, params['W3'].T)  # (batch_size, hidden_size)
    dz2 = da2 * relu_derivative(caches['z2'])  # (batch_size, hidden_size)
    if use_bn:
        dz2, dgamma2, dbeta2 = batch_norm_backward(dz2, caches['bn2_cache'])

    # 隐藏全连接层梯度
    dW2 = np.dot(caches['flattened'].T, dz2) / batch_size  # (flattened_size, hidden_size)
    db2 = np.sum(dz2, axis=0, keepdims=True).T / batch_size  # (hidden_size, 1)
    dflattened = np.dot(dz2, params['W2'].T)  # (batch_size, flattened_size)

    # 获取第1层卷积的输出通道数和空间尺寸
    num_output_channels_1 = params['W1'].shape[0]
    _, _, height_1, width_1 = caches['a1'].shape  # (batch_size, num_channels, height_1, width_1)

    # 重塑回卷积层输出形状
    da1 = dflattened.reshape(batch_size, num_output_channels_1, height_1, width_1)

    # 第1层卷积梯度 - 使用修复后的conv2dGradient组件
    if use_bn:
        da1 = da1 * relu_derivative(caches['a1'])
        da1, dgamma1, dbeta1 = batch_norm_backward(da1, caches['bn1_cache'])
    dW1 = np.zeros_like(params['W1'])
    db1 = np.zeros_like(params['b1'])

    for i in range(batch_size):
        for k in range(num_output_channels_1):
            _, kernel_grad, bias_grad = conv2dGradient(
                da1[i, k],
                caches['X_batch'][i],
                params['W1'][k],
                stride=1,
                padding=0
            )
            dW1[k] += kernel_grad
            db1[k] += bias_grad

    dW1 /= batch_size
    db1 /= batch_size

    grads = {
        'dW3': dW3,
        'db3': db3,
        'dW2': dW2,
        'db2': db2,
        'dW1': dW1,
        'db1': db1
    }
    if use_bn:
        grads['dgamma2'] = dgamma2
        grads['dbeta2'] = dbeta2
        grads['dgamma1'] = dgamma1
        grads['dbeta1'] = dbeta1

    return grads

# ==================== 4. 参数初始化 ====================

def calculate_output_size(input_size, kernel_size, stride, padding=0):
    """计算卷积输出尺寸"""
    return (input_size - kernel_size + 2 * padding) // stride + 1

def initialize_simple_parameters(input_height=64, input_width=64, hidden_size=64):
    """初始化简化CNN网络参数"""
    params = {}
    
    # 卷积层参数：2个3x3卷积核
    params['W1'] = np.random.randn(2, 3, 3) * 0.1  # 2个3x3卷积核
    params['b1'] = np.zeros((2,))
    params['gamma1'] = np.ones((2,))
    params['beta1'] = np.zeros((2,))
    
    # 动态计算全连接层输入维度
    # 第1层: input_size -> output_size (stride=1, kernel=3)
    h1 = calculate_output_size(input_height, 3, 1)
    w1 = calculate_output_size(input_width, 3, 1)
    flattened_size = 2 * h1 * w1
    
    # 隐藏层全连接参数
    params['W2'] = np.random.randn(flattened_size, hidden_size) * 0.1
    params['b2'] = np.zeros((hidden_size, 1))
    params['gamma2'] = np.ones((hidden_size,))
    params['beta2'] = np.zeros((hidden_size,))
    
    # 输出层全连接参数
    params['W3'] = np.random.randn(hidden_size, 10) * 0.1  # 10 classes
    params['b3'] = np.zeros((10, 1))
    
    return params

# ==================== 5. 预测函数 ====================

def predict(X, params, use_bn=False):
    """预测函数"""
    # 重塑输入为(batch_size, height, width)
    if X.ndim == 2:
        batch_size = X.shape[1]
        X_reshaped = X.T.reshape(batch_size, 64, 64)
    else:
        X_reshaped = X
    
    _, output = simple_cnn_forward(X_reshaped, params, use_bn=use_bn)
    predictions = np.argmax(output, axis=0)
    return predictions

def accuracy(Y_pre, Y_true):
    """准确率计算"""
    if Y_true.ndim > 1 and Y_true.shape[0] > 1:
        Y_true_flat = np.argmax(Y_true, axis=0)
    else:
        Y_true_flat = Y_true.flatten()
    
    Y_pre_flat = Y_pre.flatten()
    return np.mean(Y_pre_flat == Y_true_flat)

# ==================== 6. 训练函数 ====================

def train_simple_cnn(loader, params, learning_rate=0.001, epochs=10, use_Adam=False, use_bn=False, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """训练简化CNN模型"""
    loss_history = []
    train_acc_history = []
    test_acc_history = []
    
    # 获取测试集数据
    X_test_full, y_test_full = loader.get_test_data(one_hot=False)
    X_test_reshaped = X_test_full.T.reshape(-1, 64, 64)

    if use_Adam:
        v_dw1 = np.zeros_like(params['W1'])
        v_db1 = np.zeros_like(params['b1'])
        v_dw2 = np.zeros_like(params['W2'])
        v_db2 = np.zeros_like(params['b2'])
        v_dw3 = np.zeros_like(params['W3'])
        v_db3 = np.zeros_like(params['b3'])
        s_dw1 = np.zeros_like(params['W1'])
        s_db1 = np.zeros_like(params['b1'])
        s_dw2 = np.zeros_like(params['W2'])
        s_db2 = np.zeros_like(params['b2'])
        s_dw3 = np.zeros_like(params['W3'])
        s_db3 = np.zeros_like(params['b3'])
    
    t = 0  # Adam优化器的时间步
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        # 使用数据加载器获取所有批次
        for X_batch_flat, y_batch, batch_idx, total_batches in loader.get_all_batches():
            # 重塑输入数据为(batch_size, 64, 64)
            batch_size = X_batch_flat.shape[0]
            X_batch = X_batch_flat.reshape(batch_size, 64, 64)
            
            # 转换标签为one-hot编码
            Y_batch = one_hot_encode(y_batch, num_classes=10)
            
            # 前向传播
            caches, output = simple_cnn_forward(X_batch, params, use_bn=use_bn)
            
            # 计算损失
            loss = cross_entropy_loss(output, Y_batch)
            epoch_loss += loss
            num_batches += 1
            
            # 反向传播
            grads = simple_cnn_backward(caches, Y_batch, params, use_bn=use_bn)
            
            if use_Adam:
                t = t + 1  # Adam时间步增加
                
                # 更新Adam变量
                v_dw1 = beta1 * v_dw1 + (1 - beta1) * grads['dW1']
                v_db1 = beta1 * v_db1 + (1 - beta1) * grads['db1']
                v_dw2 = beta1 * v_dw2 + (1 - beta1) * grads['dW2']
                v_db2 = beta1 * v_db2 + (1 - beta1) * grads['db2']
                v_dw3 = beta1 * v_dw3 + (1 - beta1) * grads['dW3']
                v_db3 = beta1 * v_db3 + (1 - beta1) * grads['db3']

                s_dw1 = beta2 * s_dw1 + (1 - beta2) * grads['dW1'] ** 2
                s_db1 = beta2 * s_db1 + (1 - beta2) * grads['db1'] ** 2
                s_dw2 = beta2 * s_dw2 + (1 - beta2) * grads['dW2'] ** 2
                s_db2 = beta2 * s_db2 + (1 - beta2) * grads['db2'] ** 2
                s_dw3 = beta2 * s_dw3 + (1 - beta2) * grads['dW3'] ** 2
                s_db3 = beta2 * s_db3 + (1 - beta2) * grads['db3'] ** 2

                # 修正偏差
                v_dw1_corrected = v_dw1 / (1 - beta1 ** t)
                v_db1_corrected = v_db1 / (1 - beta1 ** t)
                v_dw2_corrected = v_dw2 / (1 - beta1 ** t)
                v_db2_corrected = v_db2 / (1 - beta1 ** t)
                v_dw3_corrected = v_dw3 / (1 - beta1 ** t)
                v_db3_corrected = v_db3 / (1 - beta1 ** t)
                s_dw1_corrected = s_dw1 / (1 - beta2 ** t)
                s_db1_corrected = s_db1 / (1 - beta2 ** t)
                s_dw2_corrected = s_dw2 / (1 - beta2 ** t)
                s_db2_corrected = s_db2 / (1 - beta2 ** t)
                s_dw3_corrected = s_dw3 / (1 - beta2 ** t)
                s_db3_corrected = s_db3 / (1 - beta2 ** t)

                # 更新参数
                params['W1'] = params['W1'] - (learning_rate / (np.sqrt(s_dw1_corrected) + epsilon)) * v_dw1_corrected
                params['b1'] = params['b1'] - (learning_rate / (np.sqrt(s_db1_corrected) + epsilon)) * v_db1_corrected
                params['W2'] = params['W2'] - (learning_rate / (np.sqrt(s_dw2_corrected) + epsilon)) * v_dw2_corrected
                params['b2'] = params['b2'] - (learning_rate / (np.sqrt(s_db2_corrected) + epsilon)) * v_db2_corrected
                params['W3'] = params['W3'] - (learning_rate / (np.sqrt(s_dw3_corrected) + epsilon)) * v_dw3_corrected
                params['b3'] = params['b3'] - (learning_rate / (np.sqrt(s_db3_corrected) + epsilon)) * v_db3_corrected
                if use_bn:
                    params['gamma1'] = params['gamma1'] - learning_rate * grads['dgamma1']
                    params['beta1'] = params['beta1'] - learning_rate * grads['dbeta1']
                    params['gamma2'] = params['gamma2'] - learning_rate * grads['dgamma2']
                    params['beta2'] = params['beta2'] - learning_rate * grads['dbeta2']

            else:
                params['W1'] = params['W1'] - learning_rate * grads['dW1']
                params['b1'] = params['b1'] - learning_rate * grads['db1']
                params['W2'] = params['W2'] - learning_rate * grads['dW2']
                params['b2'] = params['b2'] - learning_rate * grads['db2']
                params['W3'] = params['W3'] - learning_rate * grads['dW3']
                params['b3'] = params['b3'] - learning_rate * grads['db3']
                if use_bn:
                    params['gamma1'] -= learning_rate * grads['dgamma1']
                    params['beta1'] -= learning_rate * grads['dbeta1']
                    params['gamma2'] -= learning_rate * grads['dgamma2']
                    params['beta2'] -= learning_rate * grads['dbeta2']

            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                print(f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{total_batches} - Loss: {loss:.4f}")
        
        # 记录平均损失
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # 评估训练和测试准确率
        # 训练集评估（取一个小样本）
        train_sample_size = min(1000, loader.X_train.shape[0])
        train_indices = np.random.choice(loader.X_train.shape[0], train_sample_size, replace=False)
        X_train_sample = loader.X_train[train_indices].reshape(train_sample_size, 64, 64)
        y_train_sample = loader.y_train[train_indices]
        train_pred = predict(X_train_sample, params, use_bn=use_bn)
        train_acc = accuracy(train_pred, y_train_sample)
        train_acc_history.append(train_acc)
        
        # 测试集评估
        test_pred = predict(X_test_reshaped, params, use_bn=use_bn)
        test_acc = accuracy(test_pred, y_test_full)
        test_acc_history.append(test_acc)
        
        print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
    
    return loss_history, train_acc_history, test_acc_history, params

# ==================== 7. 主函数 ====================

def main():
    """主函数：训练并评估简化CNN模型"""
    
    print("="*60)
    print("简化手势数字 CNN 分类器（1个卷积层 + 隐藏层）")
    print("="*60)
    
    # 创建数据加载器，指定image_size=(100, 100)
    print("\n正在加载手势数字数据集...")
    loader = HandDigitalDataLoader(
        batch_size=128,
        shuffle=True,
        normalize=True,
        image_size=(64, 64),  # 指定输入尺寸为64x64
        grayscale=True
    )
    
    # 获取训练集信息
    info = loader.get_train_info()
    print(f"\n数据集信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 初始化网络参数
    print("\n初始化简化CNN网络参数...")
    params = initialize_simple_parameters(hidden_size=256)
    
    # 训练模型
    print("\n开始训练简化CNN模型...")
    loss_history, train_acc_history, test_acc_history, trained_params = \
        train_simple_cnn(loader, params, learning_rate=0.01, epochs=200, use_Adam=True, use_bn=True)

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