import sys
from pathlib import Path

# 把项目根目录 ai_stu 加进 sys.path，让 import KSNet 能找到包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import KSNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 设置字体（兼容macOS和Windows）
import platform
system_name = platform.system()
if system_name == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'STHeiti']
elif system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
# 解决坐标轴负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 生成数据集 ====================
# make_moons 生成一个月牙形的二分类数据，比线性可分更有挑战性
# n_samples: 样本数  noise: 噪声程度
np.random.seed(42)
X, Y = make_moons(n_samples=300, noise=0.2, random_state=42)

# X: (300, 2)  每个样本2个特征
# Y: (300,)    标签0或1

# 划分训练集和测试集 (80%训练, 20%测试)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

# 转置为 (特征数, 样本数)，方便向量化计算
# X_train = X_train.T  # (2, 240)
Y_train = Y_train.reshape(-1, 1)  # (1, 240)
# X_test = X_test.T  # (2, 60)
Y_test = Y_test.reshape(-1, 1)  # (1, 60)

n_features, m_train = X_train.shape
print(f"训练集: X{X_train.shape}, Y{Y_train.shape}")
print(f"测试集: X{X_test.shape}, Y{Y_test.shape}")


linear_model = KSNet.KSLinear(input_dim=2, output_dim=1)
layers = [linear_model]
loss = KSNet.KSBinaryLogisticLoss()
optimizer = KSNet.KSSGDOptimizer(lr=0.5, layers=layers)

losses = []  # 记录每100轮的损失

for epoch in range(2000):
    logits = linear_model.forward(X_train)
    l = loss.forward(logits, Y_train)

    dout = loss.backward()

    linear_model.backward(dout)
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 0:
        losses.append(l)
        print(f"Epoch {epoch:4d} | Loss: {l:.6f}")
