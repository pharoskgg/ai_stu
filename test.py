import numpy as np
from KSNet import KSLinear, KSSGDOptimizer, KSBinaryLogisticLoss

# 1. 搭建两层网络，二分类输出维度=1
fc1 = KSLinear(input_dim=5, output_dim=16)
fc2 = KSLinear(input_dim=16, output_dim=1)
layers = [fc1, fc2]

# 2. 损失函数 + 优化器
loss_fn = KSBinaryLogisticLoss()
opt = KSSGDOptimizer(layers, lr=0.05, weight_decay=1e-4)

# 模拟数据集
batch_size = 32
x = np.random.randn(batch_size, 5)
# 生成0/1二分类标签
y = np.random.randint(0, 2, size=(batch_size, 1)).astype(np.float32)

# 训练一轮流程
# 1. 前向传播
h1 = fc1.forward(x)
logits = fc2.forward(h1)  # logits 未sigmoid，直接丢给损失函数
loss_val = loss_fn.forward(logits, y)
print(f"Batch Loss: {loss_val:.4f}")

# 2. 反向传播
d_logits = loss_fn.backward()       # 损失输出梯度，传给最后一层Linear
d_h1 = fc2.backward(d_logits)
d_x = fc1.backward(d_h1)

# 3. 参数更新 + 清空梯度
opt.step()
opt.zero_grad()

# 推理预测（sigmoid转换概率）
def predict_prob(logits: np.ndarray):
    z = logits
    mask = z >= 0
    sig = np.empty_like(z)
    sig[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
    exp_z = np.exp(z[~mask])
    sig[~mask] = exp_z / (1.0 + exp_z)
    return sig

prob = predict_prob(logits)
print("预测概率 shape:", prob.shape)