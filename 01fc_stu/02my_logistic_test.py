import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# 设置随机种子，保证结果可复现
np.random.seed(42)

# ==================== 1. 生成数据集 ====================
# make_moons 生成一个月牙形的二分类数据，比线性可分更有挑战性
# n_samples: 样本数  noise: 噪声程度
X, Y = make_moons(n_samples=300, noise=0.2, random_state=42)

# X: (300, 2)  每个样本2个特征
# Y: (300,)    标签0或1

# 划分训练集和测试集 (80%训练, 20%测试)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

# 转置为 (特征数, 样本数)，方便向量化计算
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
W = np.random.randn(n_features, 1) * 0.01
b = 0.0

print(f"\n初始参数: W shape{W.shape}, b={b}")

# W.T:(1, 2) X:(2, m)
def forwawrd_propagation(X, W, b):
    z = np.dot(W.T, X) + b
    a = sigmoid(z)
    return a

# 适用于二分类
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# A为标签，Y为计算后的输出值，A.shape = (1, m)
def logistic_loss(A, Y):
    m = Y.shape[1]
    return -(1 / m) * np.sum(A * np.log(Y) + (1 - A) * np.log(1-Y))

# X为行列式，每列为一个样本,A为预测值
def backward_progation(X, Y, A):
    m = X.shape[1]

    dz = A - Y # (1, m)
    
    dw = 1 / m * np.dot(X, dz.T) #(2, m)  (m, 1)

    db = 1 / m * np.sum(dz)

    return dw, db

loop = 2000
loss_history = []
learn_rate = 0.5

for i in range(loop):
    A = forwawrd_propagation(X_train, W, b)

    loss = logistic_loss(Y_train, A)

    dw, db = backward_progation(X_train, Y_train, A)

    W = W - learn_rate * dw

    b = b - learn_rate * db

    if i % 100 == 0:
        loss_history.append(loss)
        print(loss)

print(W, b)

def predict(X, W, b):
    A = forwawrd_propagation(X, W, b)
    return (A >= 0.5).astype(int)

def accuracy(Y_pre, Y_ture):
    return np.mean(Y_pre == Y_ture)

Y_pred_test = predict(X_test, W, b)
test_acc = accuracy(Y_pred_test, Y_test)
print(test_acc)