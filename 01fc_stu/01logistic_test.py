# 二分类练习：使用Logistic回归（单层神经网络）
# 包含：sigmoid激活、二元交叉熵损失、梯度下降、反向传播

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 设置字体为SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决坐标轴负号显示问题
plt.rcParams['axes.unicode_minus'] = False

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


# ==================== 3. 定义核心函数 ====================

def sigmoid(z):
    """
    sigmoid激活函数
    z: 可以是标量、向量或矩阵
    输出范围 (0, 1)，表示样本属于类别1的概率
    """
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy_loss(A, Y):
    """
    二元交叉熵损失函数（对应笔记中的公式）
    A: 预测概率 (1, m)
    Y: 真实标签 (1, m)
    返回：平均损失值（标量）
    """
    m = Y.shape[1]
    # 防止log(0)出现数值问题，加入极小值epsilon
    epsilon = 1e-5
    A = np.clip(A, epsilon, 1 - epsilon)

    # L = -(1/m) * sum(y*log(a) + (1-y)*log(1-a))
    loss = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return loss


def forward_propagation(X, W, b):
    """
    前向传播
    X: 输入特征 (n_features, m)
    W: 权重 (n_features, 1)
    b: 偏置 (标量)

    返回:
      Z: 线性计算结果 W^T * X + b  (1, m)
      A: 经过sigmoid后的预测概率 (1, m)
    """
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    return Z, A


def backward_propagation(X, Y, A):
    """
    反向传播：计算梯度
    根据笔记推导:
      dZ = A - Y
      dW = (1/m) * X * dZ^T
      db = (1/m) * sum(dZ)

    X: (n_features, m)
    Y: (1, m)
    A: (1, m)

    返回: dW, db
    """
    m = X.shape[1]

    # 链式法则最终结果: dL/dZ = A - Y
    dZ = A - Y  # (1, m)

    # dW = (1/m) * X * dZ^T   => (n_features, m) @ (m, 1) = (n_features, 1)
    dW = (1 / m) * np.dot(X, dZ.T)

    # db = (1/m) * sum(dZ)
    db = (1 / m) * np.sum(dZ)

    return dW, db


# ==================== 4. 训练模型（梯度下降） ====================

learning_rate = 0.5  # 学习率
epochs = 2000        # 迭代次数

losses = []  # 记录每100轮的损失

print("\n开始训练...")

for i in range(epochs):
    # 前向传播
    Z, A = forward_propagation(X_train, W, b)

    # 计算损失
    loss = binary_cross_entropy_loss(A, Y_train)

    # 反向传播
    dW, db = backward_propagation(X_train, Y_train, A)

    # 更新参数（梯度下降）
    W = W - learning_rate * dW
    b = b - learning_rate * db

    # 每100轮打印一次
    if i % 100 == 0:
        losses.append(loss)
        print(f"Epoch {i:4d} | Loss: {loss:.6f}")

print(f"\n训练完成！最终损失: {loss:.6f}")
print(f"最终参数: W = {W.flatten()}, b = {b:.4f}")


# ==================== 5. 预测与评估 ====================

def predict(X, W, b):
    """
    预测函数
    概率 >= 0.5 判定为类别1，否则为类别0
    """
    _, A = forward_propagation(X, W, b)
    return (A >= 0.5).astype(int)


def accuracy(Y_pred, Y_true):
    """计算准确率"""
    return np.mean(Y_pred == Y_true)


# 在测试集上评估
Y_pred_test = predict(X_test, W, b)
test_acc = accuracy(Y_pred_test, Y_test)
print(f"\n测试集准确率: {test_acc * 100:.2f}%")

# 在训练集上评估
Y_pred_train = predict(X_train, W, b)
train_acc = accuracy(Y_pred_train, Y_train)
print(f"训练集准确率: {train_acc * 100:.2f}%")


# ==================== 6. 可视化 ====================

def plot_decision_boundary(X_data, Y_data, W, b, title=""):
    """
    绘制决策边界
    """
    # 确定坐标范围
    x_min, x_max = X_data[0, :].min() - 0.5, X_data[0, :].max() + 0.5
    y_min, y_max = X_data[1, :].min() - 0.5, X_data[1, :].max() + 0.5

    # 生成网格点
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # 将网格点展平并预测
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    _, A_grid = forward_propagation(grid_points, W, b)
    A_grid = A_grid.reshape(xx.shape)

    # 绘制
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, A_grid, levels=50, cmap="RdBu_r", alpha=0.6)
    plt.colorbar(label="预测为类别1的概率")
    plt.contour(xx, yy, A_grid, levels=[0.5], colors="black", linewidths=2)

    # 绘制样本点
    Y_flat = Y_data.flatten()
    plt.scatter(X_data[0, Y_flat == 0], X_data[1, Y_flat == 0],
                c="blue", edgecolors="k", label="类别 0")
    plt.scatter(X_data[0, Y_flat == 1], X_data[1, Y_flat == 1],
                c="red", edgecolors="k", label="类别 1")

    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt


# 绘制训练集结果
plt1 = plot_decision_boundary(X_train, Y_train, W, b,
                              f"训练集决策边界 (准确率: {train_acc * 100:.1f}%)")
plt1.savefig("train_boundary.png")
plt1.show()
print("已保存: train_boundary.png")

# 绘制测试集结果
plt2 = plot_decision_boundary(X_test, Y_test, W, b,
                              f"测试集决策边界 (准确率: {test_acc * 100:.1f}%)")
plt2.savefig("test_boundary.png")
plt2.show()
print("已保存: test_boundary.png")

# 绘制损失下降曲线
plt.figure(figsize=(8, 5))
plt.plot(range(0, epochs, 100), losses, marker="o", linewidth=2)
plt.xlabel("训练轮次 (Epoch)")
plt.ylabel("损失值 (Loss)")
plt.title("损失函数随训练下降曲线")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"loss_curve.png")
plt.show()
print("已保存: loss_curve.png")

print("\n练习完成！请对照笔记中的公式理解每一行代码。")
