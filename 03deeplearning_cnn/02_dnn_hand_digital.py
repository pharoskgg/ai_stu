"""
基于MNIST数据集的深度神经网络(DNN)分类器

功能说明:
1. 使用mnist_dataloader加载MNIST手写数字数据集
2. 实现多层全连接神经网络
3. 支持多种激活函数(ReLU, Leaky ReLU, Tanh)
4. 使用Softmax输出层进行多分类
5. 可视化训练过程和结果

网络结构:
输入层(784) -> 隐藏层1(h1) -> 隐藏层2(h2) -> 输出层(10) -> Softmax
"""

import numpy as np
import matplotlib.pyplot as plt
from hand_digital_dataloader import HandDigitalDataLoader
import warnings

# 过滤数值警告
warnings.filterwarnings('ignore')

# 设置随机种子，保证结果可复现
np.random.seed(42)



# ==================== 3. 激活函数定义 ====================
def softmax(z):
    """Softmax激活函数（输出层）- 数值稳定版本"""
    # 减去最大值以防止数值溢出
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def sigmoid(z):
    """Sigmoid激活函数（仅用于二分类对比）"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    """Sigmoid的导数"""
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    """ReLU激活函数"""
    return np.maximum(0, z)

def relu_derivative(a):
    """ReLU的导数"""
    return (a > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU激活函数"""
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(a, alpha=0.01):
    """Leaky ReLU的导数"""
    return np.where(a > 0, 1.0, alpha)

def tanh(z):
    """Tanh激活函数"""
    return np.tanh(z)

def tanh_derivative(a):
    """Tanh的导数: 1 - a^2"""
    return 1 - a ** 2

# 激活函数映射表
ACTIVATION_FUNCTIONS = {
    'relu': {'func': relu, 'deriv': relu_derivative},
    'leaky_relu': {'func': leaky_relu, 'deriv': leaky_relu_derivative},
    'tanh': {'func': tanh, 'deriv': tanh_derivative},
    'sigmoid': {'func': sigmoid, 'deriv': sigmoid_derivative}
}

global dropout_mask  # 定义全局变量用于存储dropout掩码

# ==================== 4. 神经网络核心函数 ====================
def forward_propagation(X, w1, b1, w2, b2, activation='tanh', use_softmax=False, keep_prob=1.0):
    """前向传播
    
    参数:
        X: 输入数据 (n_features, m)
        w1, b1: 隐藏层权重和偏置
        w2, b2: 输出层权重和偏置
        activation: 隐藏层激活函数类型
        use_softmax: 是否使用softmax作为输出层（多分类）
        keep_prob: Dropout保持率
    
    返回:
        z1, a1, z2, a2, D1: 各层的线性输出和激活输出
    """
    act_func = ACTIVATION_FUNCTIONS[activation]['func']
    
    # 隐藏层
    z1 = np.dot(w1.T, X) + b1
    a1 = act_func(z1)

    # Dropout（仅隐藏层，keep_prob < 1 时启用）
    D1 = None
    if keep_prob < 1.0:
        D1 = np.random.rand(a1.shape[0], a1.shape[1]) < keep_prob
        a1 *= D1
        a1 /= keep_prob

    # 输出层
    z2 = np.dot(w2.T, a1) + b2
    
    if use_softmax:
        # 多分类：使用softmax
        a2 = softmax(z2)
    else:
        # 二分类：使用sigmoid
        a2 = sigmoid(z2)

    return z1, a1, z2, a2, D1

def cross_entropy_loss(A, Y):
    """交叉熵损失函数（适用于softmax多分类）
    
    参数:
        A: 预测概率 (n_classes, m)
        Y: 真实标签one-hot编码 (n_classes, m)
    
    返回:
        loss: 平均交叉熵损失
    """
    m = Y.shape[1]
    epsilon = 1e-8
    # 防止log(0)
    loss = -(1 / m) * np.sum(Y * np.log(A + epsilon))
    return loss

def logistic_loss(A, Y):
    """逻辑回归损失函数（仅用于二分类对比）"""
    m = Y.shape[1]
    epsilon = 1e-8
    return -(1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))

def backward_propagation(X, Y, W2, z1, a1, z2, a2, activation='tanh', use_softmax=False, D1=None, keep_prob=1.0):
    """反向传播
    
    参数:
        X: 输入数据 (n_features, m)
        Y: 真实标签 (n_classes, m)
        W2: 输出层权重
        z1, a1: 隐藏层的线性输出和激活输出
        z2, a2: 输出层的线性输出和激活输出
        activation: 隐藏层激活函数类型
        use_softmax: 是否使用softmax
        D1: Dropout掩码
        keep_prob: Dropout保持率

    返回:
        dw1, db1, dw2, db2: 各参数的梯度
    """
    act_deriv = ACTIVATION_FUNCTIONS[activation]['deriv']
    m = X.shape[1]

    if use_softmax:
        # Softmax + Cross Entropy 的梯度简化形式
        dz2 = a2 - Y  # (n_classes, m)
    else:
        # Sigmoid + Binary Cross Entropy
        dz2 = a2 - Y  # (1, m)
    
    dw2 = (1 / m) * np.dot(a1, dz2.T)  # (h, n_classes)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)  # (n_classes, 1)
    
    act_derivative = act_deriv(a1)
    dz1 = np.dot(W2, dz2) * act_derivative  # (h, m)
    # Dropout梯度：被杀死的神经元梯度归零
    if D1 is not None:
        dz1 *= D1 / keep_prob

    dw1 = (1 / m) * np.dot(X, dz1.T)  # (n_features, h)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)  # (h, 1)

    return dw1, db1, dw2, db2

def predict(X, w1, b1, w2, b2, activation='tanh', use_softmax=False):
    """预测函数
    
    参数:
        X: 输入数据
        w1, b1, w2, b2: 网络参数
        activation: 隐藏层激活函数
        use_softmax: 是否使用softmax输出
    
    返回:
        predictions: 预测的类别标签（非one-hot）
    """
    _, _, _, A, _ = forward_propagation(X, w1, b1, w2, b2, activation=activation, use_softmax=use_softmax)
    
    if use_softmax:
        # 多分类：取概率最大的类别
        return np.argmax(A, axis=0)
    else:
        # 二分类：阈值判断
        return (A >= 0.5).astype(int)

def accuracy(Y_pre, Y_true):
    """准确率计算
    
    参数:
        Y_pre: 预测标签 (m,) 或 (1, m)
        Y_true: 真实标签 (m,) 或 (n_classes, m) one-hot
    
    返回:
        acc: 准确率
    """
    # 如果Y_true是one-hot编码，转换为类别索引
    if Y_true.ndim > 1 and Y_true.shape[0] > 1:
        Y_true_flat = np.argmax(Y_true, axis=0)
    else:
        Y_true_flat = Y_true.flatten()
    
    # 确保Y_pre是一维数组
    Y_pre_flat = Y_pre.flatten()
    
    return np.mean(Y_pre_flat == Y_true_flat)


def one_hot_encode(y, num_classes):
    """将标签转换为one-hot编码
    
    参数:
        y: 标签数组 (m,)
        num_classes: 类别总数
    
    返回:
        one_hot: one-hot编码后的标签 (num_classes, m)
    """
    m = y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[y, np.arange(m)] = 1
    return one_hot

def train(X_train, y_train, X_test, y_test, hidden_size=128, activation='relu', keep_prob=1.0, use_Adam=False, beta1=0.9, beta2=0.999, learn_rate=0.01, epochs=20, epsilon=1e-8):
    """训练DNN模型"""
    # 将标签转换为one-hot编码
    Y_train = one_hot_encode(y_train, num_classes=10)
    Y_test = one_hot_encode(y_test, num_classes=10)

    n_features = X_train.shape[0]
    n_classes = Y_train.shape[0]

    # 初始化权重和偏置
    w1 = np.random.randn(n_features, hidden_size) * 0.1
    b1 = np.zeros((hidden_size, 1))
    w2 = np.random.randn(hidden_size, n_classes) * 0.1
    b2 = np.zeros((n_classes, 1))

    if use_Adam:
        v_dw1 = np.zeros_like(w1)
        v_db1 = np.zeros_like(b1)
        v_dw2 = np.zeros_like(w2)
        v_db2 = np.zeros_like(b2)
        s_dw1 = np.zeros_like(w1)
        s_db1 = np.zeros_like(b1)
        s_dw2 = np.zeros_like(w2)
        s_db2 = np.zeros_like(b2)

    loss_history = []
    train_acc_history = []
    test_acc_history = []

    t = 0  # Adam优化器的时间步计数器
    # 训练循环
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        # 使用minibatch训练
        for i in range(0, X_train.shape[1], 128):
            X_batch = X_train[:, i:i+128]
            Y_batch = Y_train[:, i:i+128]

            # 前向传播
            z1, a1, z2, a2, D1 = forward_propagation(X_batch, w1, b1, w2, b2, activation=activation, use_softmax=True, keep_prob=keep_prob)
            
            # 计算损失
            loss = cross_entropy_loss(a2, Y_batch)
            epoch_loss += loss
            num_batches += 1
            
            # 反向传播
            dw1, db1, dw2, db2 = backward_propagation(X_batch, Y_batch, w2, z1, a1, z2, a2, activation=activation, use_softmax=True, D1=D1, keep_prob=keep_prob)
            
            # 更新参数
            if use_Adam:
                t += 1  # 增加时间步计数器
                
                # 更新Adam变量
                v_dw1 = beta1 * v_dw1 + (1 - beta1) * dw1
                v_db1 = beta1 * v_db1 + (1 - beta1) * db1
                v_dw2 = beta1 * v_dw2 + (1 - beta1) * dw2
                v_db2 = beta1 * v_db2 + (1 - beta1) * db2

                s_dw1 = beta2 * s_dw1 + (1 - beta2) * dw1 ** 2
                s_db1 = beta2 * s_db1 + (1 - beta2) * db1 ** 2
                s_dw2 = beta2 * s_dw2 + (1 - beta2) * dw2 ** 2
                s_db2 = beta2 * s_db2 + (1 - beta2) * db2 ** 2

                # 修正偏差
                v_dw1_corrected = v_dw1 / (1 - beta1 ** t)
                v_db1_corrected = v_db1 / (1 - beta1 ** t)
                v_dw2_corrected = v_dw2 / (1 - beta1 ** t)
                v_db2_corrected = v_db2 / (1 - beta1 ** t)
                s_dw1_corrected = s_dw1 / (1 - beta2 ** t)
                s_db1_corrected = s_db1 / (1 - beta2 ** t)
                s_dw2_corrected = s_dw2 / (1 - beta2 ** t)
                s_db2_corrected = s_db2 / (1 - beta2 ** t)

                # 更新参数
                w1 = w1 - (learn_rate / (np.sqrt(s_dw1_corrected) + epsilon)) * v_dw1_corrected
                b1 = b1 - (learn_rate / (np.sqrt(s_db1_corrected) + epsilon)) * v_db1_corrected
                w2 = w2 - (learn_rate / (np.sqrt(s_dw2_corrected) + epsilon)) * v_dw2_corrected
                b2 = b2 - (learn_rate / (np.sqrt(s_db2_corrected) + epsilon)) * v_db2_corrected

            else:
                w1 = w1 - learn_rate * dw1
                b1 = b1 - learn_rate * db1
                w2 = w2 - learn_rate * dw2
                b2 = b2 - learn_rate * db2
            print(f"Epoch {epoch + 1}/{epochs} - Batch {i // 128 + 1} - Loss: {loss:.4f}")
        
        # 记录平均损失
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # 每5个epoch评估一次模型性能
        if (epoch + 1) % 1 == 0:
            train_pred = predict(X_train, w1, b1, w2, b2, activation=activation, use_softmax=True)
            test_pred = predict(X_test, w1, b1, w2, b2, activation=activation, use_softmax=True)
            train_acc = accuracy(train_pred, y_train)
            test_acc = accuracy(test_pred, y_test)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")

    return loss_history, train_acc_history, test_acc_history, w1, b1, w2, b2

def save_model(w1, b1, w2, b2, filename):
    """保存模型参数到文件"""
    np.savez(filename, w1=w1, b1=b1, w2=w2, b2=b2)
    print(f"模型参数已保存到 {filename}")

# ==================== 6. 主函数 ====================
def main():
    """主函数：训练并评估DNN模型"""
    
    print("="*60)
    print("手写数字 DNN 分类器")
    print("="*60)
    
    # 创建数据加载器
    print("\n正在加载手写数字数据集...")
    loader = HandDigitalDataLoader(
        batch_size=128,
        shuffle=True,
        normalize=True,
        # image_size=(50, 50),
        grayscale = True
    )
    
    # 获取训练集信息
    info = loader.get_train_info()
    print(f"\n数据集信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 数据处理，拉平成一维数据
    X_train = loader.X_train.reshape(loader.X_train.shape[0], -1).T  # (784, 60000)
    y_train = loader.y_train  # (60000,)
    X_test = loader.X_test.reshape(loader.X_test.shape[0], -1).T
    y_test = loader.y_test

    # 训练模型
    loss_history, train_acc_history, test_acc_history, w1, b1, w2, b2 = \
    train(X_train, y_train, X_test, y_test, hidden_size=1024, 
          activation='relu', keep_prob=0.8, learn_rate=0.001, epochs=200, use_Adam=True)

    save_model(w1, b1, w2, b2, 'dnn_hand_digit_model_grayscale_Adam_keep_prob_0.8.npz')

    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
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
