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

# ==================== 2. CNN前向传播 ====================

def cnn_forward(X_batch, params):
    """
    CNN前向传播（无池化层）
    
    参数:
        X_batch: 输入批次数据 (batch_size, height, width)
        params: 网络参数字典
        
    返回:
        caches: 缓存各层的中间结果用于反向传播
        output: 最终输出 (n_classes, batch_size)
    """
    batch_size = X_batch.shape[0]
    
    # 第1层卷积: 100x100 -> 50x50 (使用3x3卷积核，stride=2, padding=0)
    W1 = params['W1']  # (32, 3, 3) - 32个3x3卷积核
    b1 = params['b1']  # (32,)
    
    conv1_outputs = []
    for i in range(batch_size):
        conv1_sample = []
        for k in range(W1.shape[0]):  # 对每个卷积核
            conv_result = convolution2d(X_batch[i], W1[k], padding=0, stride=2) + b1[k]
            conv1_sample.append(conv_result)
        conv1_outputs.append(np.array(conv1_sample))  # (32, 50, 50)
    
    conv1_outputs = np.array(conv1_outputs)  # (batch_size, 32, 50, 50)
    a1 = relu(conv1_outputs)
    
    # 第2层卷积: 50x50 -> 24x24 (使用3x3卷积核，stride=2, padding=0)
    W2 = params['W2']  # (64, 32, 3, 3) - 64个3x3卷积核
    b2 = params['b2']  # (64,)
    
    conv2_outputs = []
    for i in range(batch_size):
        conv2_sample = []
        for k in range(W2.shape[0]):  # 对每个输出通道
            conv_result = np.zeros((24, 24))
            for c in range(W2.shape[1]):  # 对每个输入通道
                conv_result += convolution2d(a1[i, c], W2[k, c], padding=0, stride=2)
            conv_result += b2[k]
            conv2_sample.append(conv_result)
        conv2_outputs.append(np.array(conv2_sample))  # (64, 24, 24)
    
    conv2_outputs = np.array(conv2_outputs)  # (batch_size, 64, 24, 24)
    a2 = relu(conv2_outputs)
    
    # 第3层卷积: 24x24 -> 11x11 (使用3x3卷积核，stride=2, padding=0)
    W3 = params['W3']  # (128, 64, 3, 3) - 128个3x3卷积核
    b3 = params['b3']  # (128,)
    
    conv3_outputs = []
    for i in range(batch_size):
        conv3_sample = []
        for k in range(W3.shape[0]):
            conv_result = np.zeros((11, 11))
            for c in range(W3.shape[1]):
                conv_result += convolution2d(a2[i, c], W3[k, c], padding=0, stride=2)
            conv_result += b3[k]
            conv3_sample.append(conv_result)
        conv3_outputs.append(np.array(conv3_sample))  # (128, 11, 11)
    
    conv3_outputs = np.array(conv3_outputs)  # (batch_size, 128, 11, 11)
    a3 = relu(conv3_outputs)
    
    # 第4层卷积: 11x11 -> 5x5 (使用3x3卷积核，stride=2, padding=0)
    W4 = params['W4']  # (256, 128, 3, 3) - 256个3x3卷积核
    b4 = params['b4']  # (256,)
    
    conv4_outputs = []
    for i in range(batch_size):
        conv4_sample = []
        for k in range(W4.shape[0]):
            conv_result = np.zeros((5, 5))
            for c in range(W4.shape[1]):
                conv_result += convolution2d(a3[i, c], W4[k, c], padding=0, stride=2)
            conv_result += b4[k]
            conv4_sample.append(conv_result)
        conv4_outputs.append(np.array(conv4_sample))  # (256, 5, 5)
    
    conv4_outputs = np.array(conv4_outputs)  # (batch_size, 256, 5, 5)
    a4 = relu(conv4_outputs)
    
    # 拉平: (batch_size, 256, 5, 5) -> (batch_size, 6400)
    flattened = a4.reshape(batch_size, -1)  # (batch_size, 6400)
    
    # 第1层全连接: 6400 -> 512
    W5 = params['W5']  # (6400, 512)
    b5 = params['b5']  # (512, 1)
    z5 = np.dot(flattened, W5) + b5.T  # (batch_size, 512)
    a5 = relu(z5.T)  # (512, batch_size)
    
    # 第2层全连接: 512 -> 10
    W6 = params['W6']  # (512, 10)
    b6 = params['b6']  # (10, 1)
    z6 = np.dot(a5.T, W6) + b6.T  # (batch_size, 10)
    a6 = softmax(z6.T)  # (10, batch_size)
    
    # 缓存中间结果用于反向传播（移除所有pool相关的缓存）
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
        'z5': z5.T,
        'a5': a5,
        'z6': z6.T,
        'a6': a6
    }
    
    return caches, a6

# ==================== 3. CNN反向传播 ====================

def cnn_backward(caches, Y, params):
    """
    CNN反向传播（无池化层）
    
    参数:
        caches: 前向传播缓存
        Y: 真实标签 (n_classes, batch_size)
        params: 网络参数
        
    返回:
        grads: 各参数的梯度
    """
    batch_size = caches['X_batch'].shape[0]
    n_classes = Y.shape[0]
    
    # 输出层梯度 (softmax + cross entropy)
    dz6 = caches['a6'] - Y  # (10, batch_size)
    
    # 全连接层2梯度
    da5 = np.dot(params['W6'], dz6)  # (512, batch_size)
    dW6 = np.dot(caches['a5'], dz6.T) / batch_size  # (512, 10)
    db6 = np.sum(dz6, axis=1, keepdims=True) / batch_size  # (10, 1)
    
    # ReLU梯度
    dz5 = da5 * relu_derivative(caches['a5'])  # (512, batch_size)
    
    # 全连接层1梯度
    dflattened = np.dot(params['W5'], dz5).T  # (batch_size, 6400)
    dW5 = np.dot(caches['flattened'].T, dz5.T) / batch_size  # (6400, 512)
    db5 = np.sum(dz5, axis=1, keepdims=True) / batch_size  # (512, 1)
    
    # 重塑回卷积层输出形状
    da4 = dflattened.reshape(batch_size, 256, 5, 5)  # (batch_size, 256, 5, 5)
    
    # 第4层卷积梯度 - 使用已有的conv2dGradient组件
    dW4 = np.zeros_like(params['W4'])
    db4 = np.zeros_like(params['b4'])
    
    for i in range(batch_size):
        for k in range(256):  # 输出通道
            bias_grad_sum = 0
            for c in range(128):  # 输入通道
                # 使用已有的反向传播组件
                # conv2dGradient(outGradent, input, kernel, stride=1, padding=0)
                input_grad, kernel_grad, bias_grad = conv2dGradient(
                    da4[i, k],           # 输出梯度 (5, 5)
                    caches['a3'][i, c],  # 输入 (11, 11)  
                    params['W4'][k, c],  # 卷积核 (3, 3)
                    stride=2,            # stride=2
                    padding=0
                )
                dW4[k, c] += kernel_grad
                bias_grad_sum += bias_grad
            db4[k] += bias_grad_sum
    
    # 平均梯度
    dW4 /= batch_size
    db4 /= batch_size

        

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

def initialize_parameters():
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
    
    # 全连接层参数（更新输入维度为6400）
    params['W5'] = np.random.randn(6400, 512) * 0.1  # 6400 -> 512
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