# 一个隐藏层
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# 设置随机种子,保证结果可复现
np.random.seed(42)

# ==================== 1. 生成数据集 ====================
# make_moons 生成一个月牙形的二分类数据,比线性可分更有挑战性
# n_samples: 样本数  noise: 噪声程度
X, Y = make_moons(n_samples=300, noise=0.2, random_state=42)

# X: (300, 2)  每个样本2个特征
# Y: (300,)    标签0或1

# 划分训练集和测试集 (80%训练, 20%测试)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

# 转置为 (特征数, 样本数),方便向量化计算
X_train = X_train.T  # (2, 240)
Y_train = Y_train.reshape(1, -1)  # (1, 240)
X_test = X_test.T  # (2, 60)
Y_test = Y_test.reshape(1, -1)  # (1, 60)

n_features, m_train = X_train.shape
print(f"训练集: X{X_train.shape}, Y{Y_train.shape}")
print(f"测试集: X{X_test.shape}, Y{Y_test.shape}")


# ==================== 2. 初始化参数 ====================
# W: (n_features, 1)  权重向量
# b: 标量  偏置
# h: 神经单元个数
h = 4
w1 = np.random.randn(n_features, h) * 0.01
b1 = np.zeros((h, 1))

w2 = np.random.randn(h, 1) * 0.01 #(h, 1)
b2 = np.zeros((1, 1))

# print(f"\n初始参数: W shape{W.shape}, b={b}")

# ==================== 3. 激活函数定义 ====================
def sigmoid(z):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Sigmoid的导数: a * (1 - a)"""
    return a * (1 - a)

def tanh(z):
    """Tanh激活函数"""
    return np.tanh(z)

def tanh_derivative(a):
    """Tanh的导数: 1 - a^2"""
    return 1 - a ** 2

def relu(z):
    """ReLU激活函数"""
    return np.maximum(0, z)

def relu_derivative(a):
    """ReLU的导数"""
    return (a > 0).astype(float)

# 激活函数映射表
ACTIVATION_FUNCTIONS = {
    'sigmoid': {'func': sigmoid, 'deriv': sigmoid_derivative},
    'tanh': {'func': tanh, 'deriv': tanh_derivative},
    'relu': {'func': relu, 'deriv': relu_derivative}
}

# W.T:(1, 2) X:(2, m)
def forward_propagation(X, w1, b1, w2, b2, activation='tanh'):
    """前向传播
    
    Args:
        X: 输入数据 (n_features, m)
        w1, b1: 第一层权重和偏置
        w2, b2: 第二层权重和偏置
        activation: 隐藏层激活函数名称 ('sigmoid', 'tanh', 'relu')
    
    Returns:
        z1, a1, a2: 中间变量和输出
    """
    if activation not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"不支持的激活函数: {activation}, 可选: {list(ACTIVATION_FUNCTIONS.keys())}")
    
    act_func = ACTIVATION_FUNCTIONS[activation]['func']
    
    # 隐藏层
    z1 = np.dot(w1.T, X) + b1 # (h, 2) (2, m) = (h, m)
    a1 = act_func(z1) #(h, m)

    #输出层
    z2 = np.dot(w2.T, a1) + b2 # (1, h) (h, m) = (1, m)
    a2 = sigmoid(z2) # (1, m)

    return z1, a1, a2

# A为预测值,Y为标签,A.shape = (1, m)
def logistic_loss(A, Y):
    m = Y.shape[1]
    # 添加epsilon防止log(0)
    epsilon = 1e-8
    return -(1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))

# X为行列式,每列为一个样本,A为预测值
def backward_propagation(X, Y, W2, z1, a1, a2, activation='tanh'):
    """反向传播
    
    Args:
        X: 输入数据
        Y: 标签
        W2: 第二层权重
        z1, a1: 第一层的中间变量
        a2: 输出
        activation: 隐藏层激活函数名称
    
    Returns:
        dw1, db1, dw2, db2: 梯度
    """
    if activation not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"不支持的激活函数: {activation}, 可选: {list(ACTIVATION_FUNCTIONS.keys())}")
    
    act_deriv = ACTIVATION_FUNCTIONS[activation]['deriv']
    
    m = X.shape[1]

    dz2 = a2 - Y # (1, m)

    dw2 = (1 / m) * np.dot(a1, dz2.T) # (h, m)(m, 1) = (h, 1)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True) # (1, 1) 使用keepdims保持维度
    
    # 使用指定激活函数的导数
    act_derivative = act_deriv(a1) # (h, m)

    dz1 = np.dot(W2, dz2) * act_derivative #(h, 1) (1, m) * (h, m) = (h, m)

    dw1 = (1 / m) * np.dot(X, dz1.T) # (2, m) (m, h) = (2, h)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True) # (h, 1) 使用keepdims保持维度

    return dw1, db1, dw2, db2

loop = 100000
loss_history = []
learn_rate = 0.04  # 提高学习率

# 设置激活函数 (可选: 'sigmoid', 'tanh', 'relu')
activation_type = 'tanh'
print(f"\n使用激活函数: {activation_type}\n")

for i in range(loop):
    z1, a1, a2 = forward_propagation(X_train, w1, b1, w2, b2, activation=activation_type)

    loss = logistic_loss(a2, Y_train)  # 修正参数顺序

    dw1, db1, dw2, db2 = backward_propagation(X_train, Y_train, w2, z1, a1, a2, activation=activation_type)

    w1 = w1 - learn_rate * dw1
    b1 = b1 - learn_rate * db1  # db1 (h, 1)

    w2 = w2 - learn_rate * dw2
    b2 = b2 - learn_rate * db2  # db2 (1, 1)

    if i % 100 == 0:
        loss_history.append(loss)
        print(f"Iteration {i}, Loss: {loss:.6f}")

# print(w1, b1, w2, b2)

def predict(X, w1, b1, w2, b2, activation='tanh'):
    _, _, A = forward_propagation(X, w1, b1, w2, b2, activation=activation)
    return (A >= 0.5).astype(int)

def accuracy(Y_pre, Y_ture):
    return np.mean(Y_pre == Y_ture)

Y_pred_test = predict(X_test, w1, b1, w2, b2, activation=activation_type)
test_acc = accuracy(Y_pred_test, Y_test)
print("测试集的准确率:{:.2f}%".format(test_acc * 100))

Y_pred_train = predict(X_train, w1, b1, w2, b2, activation=activation_type)
train_acc = accuracy(Y_pred_train, Y_train)
print("训练集的准确率:{:.2f}%".format(train_acc * 100))
