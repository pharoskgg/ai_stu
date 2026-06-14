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

# ==================== 2. CNN前向传播 ====================

def cnn_forward(X_batch, params):
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
    conv1_outputs = convolution2d_multi_channel_batch(X_batch, W1, padding=0, stride=2)
    a1 = relu(conv1_outputs + b1.reshape(1, -1, 1, 1))

    # 第2层卷积: 50x50 -> 24x24 (使用3x3卷积核，stride=2, padding=0)
    W2 = params['W2']
    b2 = params['b2']
    conv2_outputs = convolution2d_multi_channel_batch(a1, W2, padding=0, stride=2)
    a2 = relu(conv2_outputs + b2.reshape(1, -1, 1, 1))

    # 第3层卷积: 24x24 -> 11x11 (使用3x3卷积核，stride=2, padding=0)
    W3 = params['W3']
    b3 = params['b3']
    conv3_outputs = convolution2d_multi_channel_batch(a2, W3, padding=0, stride=2)
    a3 = relu(conv3_outputs + b3.reshape(1, -1, 1, 1))

    # 第4层卷积: 11x11 -> 5x5 (使用3x3卷积核，stride=2, padding=0)
    W4 = params['W4']
    b4 = params['b4']
    conv4_outputs = convolution2d_multi_channel_batch(a3, W4, padding=0, stride=2)
    a4 = relu(conv4_outputs + b4.reshape(1, -1, 1, 1))

    # 拉平: (batch_size, 256, output_h, output_w) -> (batch_size, flattened_size)
    flattened = a4.reshape(batch_size, -1)

    # 第1层全连接: flattened_size -> 512
    W5 = params['W5']
    b5 = params['b5']
    z5 = np.dot(flattened, W5) + b5.T
    a5 = relu(z5)

    # 第2层全连接: 512 -> 10
    W6 = params['W6']
    b6 = params['b6']
    z6 = np.dot(a5, W6) + b6.T
    a6 = softmax(z6)

    # 缓存中间结果用于反向传播
    caches = {
        'X_batch': X_batch,
        'conv1_outputs': conv1_outputs,
        'a1': a1,
        'conv2_outputs': conv2_outputs,
        'a2': a2,
        'conv3_outputs': conv3_outputs,
        'a3': a3,
        'conv4_outputs': conv4_outputs,
        'a4': a4,
        'flattened': flattened,
        'z5': z5,
        'a5': a5,
        'z6': z6,
        'a6': a6
    }

    return caches, a6

# ==================== 3. CNN反向传播 ====================

def cnn_backward(caches, Y, params):
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
    dz5 = da5 * relu_derivative(caches['z5'])  # (batch_size, 512)

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

    # 第4层卷积梯度 - 批次级别矩阵化计算
    _, dW4, db4 = conv2dGradient_batch(
        da4,
        caches['a3'],
        params['W4'],
        stride=2,
        padding=0
    )

    grads = {
        'dW6': dW6,
        'db6': db6,
        'dW5': dW5,
        'db5': db5,
        'dW4': dW4,
        'db4': db4
    }

    return grads

# ==================== 4. 参数初始化 ====================

def calculate_output_size(input_size, kernel_size, stride, padding=0):
    """计算卷积输出尺寸"""
    return (input_size - kernel_size + 2 * padding) // stride + 1

def initialize_parameters(input_height=100, input_width=100):
    """初始化CNN网络参数（无池化层）"""
    params = {}
    
    # 卷积层参数
    params['W1'] = np.random.randn(32, 3, 3) * 0.1  # 32个3x3卷积核
    params['b1'] = np.zeros((32,))
    
    params['W2'] = np.random.randn(64, 32, 3, 3) * 0.1  # 64个3x3卷积核
    params['b2'] = np.zeros((64,))
    
    params['W3'] = np.random.randn(128, 64, 3, 3) * 0.1  # 128个3x3卷积核
    params['b3'] = np.zeros((128,))
    
    params['W4'] = np.random.randn(256, 128, 3, 3) * 0.1  # 256个3x3卷积核
    params['b4'] = np.zeros((256,))
    
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
    
    params['W6'] = np.random.randn(512, 10) * 0.1  # 512 -> 10
    params['b6'] = np.zeros((10, 1))
    
    return params

# ==================== 5. 预测函数 ====================

def predict(X, params):
    """预测函数"""
    # 重塑输入为(batch_size, height, width)
    if X.ndim == 2:
        batch_size = X.shape[1]
        X_reshaped = X.T.reshape(batch_size, 100, 100)
    else:
        X_reshaped = X

    _, output = cnn_forward(X_reshaped, params)
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

def train_cnn(loader, params, learning_rate=0.001, epochs=10):
    """训练CNN模型（无池化层）"""
    loss_history = []
    train_acc_history = []
    test_acc_history = []
    
    # 获取测试集数据
    X_test_full, y_test_full = loader.get_test_data(one_hot=False)
    X_test_reshaped = X_test_full.T.reshape(-1, 100, 100)
    
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
            caches, output = cnn_forward(X_batch, params)
            
            # 计算损失
            loss = cross_entropy_loss(output, Y_batch)
            epoch_loss += loss
            num_batches += 1
            
            # 反向传播（简化版）
            grads = cnn_backward(caches, Y_batch, params)
            
            # 更新参数（只更新全连接层和最后一层卷积层作为示例）
            params['W6'] -= learning_rate * grads['dW6']
            params['b6'] -= learning_rate * grads['db6']
            params['W5'] -= learning_rate * grads['dW5']
            params['b5'] -= learning_rate * grads['db5']
            params['W4'] -= learning_rate * grads['dW4']
            params['b4'] -= learning_rate * grads['db4']
            
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
        train_cnn(loader, params, learning_rate=0.001, epochs=5)

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