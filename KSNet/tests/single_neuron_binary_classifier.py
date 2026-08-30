import sys
from pathlib import Path

# 把项目根目录 ai_stu 加进 sys.path，让 import KSNet 能找到包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import KSNet
import numpy as np
from KSNet.util import generate_moons_dataset, setup_chinese_font

setup_chinese_font()

(X_train, Y_train), (X_test, Y_test) = generate_moons_dataset(n_samples=300, noise=0.2, random_state=42, train_ratio=0.8)

# 转置为 (特征数, 样本数)，方便向量化计算
Y_train = Y_train.reshape(-1, 1)  # (1, 240)
Y_test = Y_test.reshape(-1, 1)  # (1, 60)

n_features, m_train = X_train.shape
print(f"训练集: X{X_train.shape}, Y{Y_train.shape}")
print(f"测试集: X{X_test.shape}, Y{Y_test.shape}")

linear_model = KSNet.KSLinear(input_dim=2, output_dim=1)
layers = [linear_model]
loss = KSNet.KSBinaryLogisticLoss()
optimizer = KSNet.KSSGDOptimizer(linear_model.parameters(), lr=0.5)

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

Y_pred_test = KSNet.KSSigmoid.apply(linear_model.forward(X_test))
Y_pred_test = (Y_pred_test > 0.5).astype(int)

Y_pred_train = KSNet.KSSigmoid.apply(linear_model.forward(X_train))
Y_pred_train = (Y_pred_train > 0.5).astype(int)

print(f"测试集准确率: {(Y_test == Y_pred_test).mean():.2%}")
print(f"训练集准确率: {(Y_train == Y_pred_train).mean():.2%}")
