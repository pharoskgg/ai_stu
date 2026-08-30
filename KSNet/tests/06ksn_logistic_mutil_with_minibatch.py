import sys
from pathlib import Path

# 把项目根目录 ai_stu 加进 sys.path，让 import KSNet 能找到包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import KSNet
import numpy as np
import matplotlib.pyplot as plt
from KSNet.util import generate_moons_dataset


def generate_mini_batches(X, Y, mini_batch_size, seed):
    """将数据随机打乱后按 mini_batch_size 切分为多个 mini-batch

    一轮(epoch)内每个样本只出现一次，抽过的不再放回；
    剩余数量不足 mini_batch_size 时返回剩余数据；
    下一轮调用时重新打乱，样本可再次被抽中。

    注意：KSNet 采用 (样本数, 特征数) 的行主序布局，
    与 02deeplearning_improve 中的 (特征数, 样本数) 列主序不同。
    """
    np.random.seed(seed)
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    mini_batches = []
    num_complete = m // mini_batch_size
    for k in range(num_complete):
        start = k * mini_batch_size
        end = start + mini_batch_size
        mini_batches.append((X_shuffled[start:end], Y_shuffled[start:end]))

    # 剩余不足 mini_batch_size 的样本
    if m % mini_batch_size != 0:
        start = num_complete * mini_batch_size
        mini_batches.append((X_shuffled[start:], Y_shuffled[start:]))

    return mini_batches


(X_train, Y_train), (X_test, Y_test) = generate_moons_dataset(n_samples=600)
# reshape 为 (样本数, 1)，匹配 KSNet 的 (batch, dim) 二维布局
Y_train = Y_train.reshape(-1, 1)  # (480, 1)
Y_test = Y_test.reshape(-1, 1)  # (120, 1)

X_train_mean = np.mean(X_train, axis=0, keepdims=True)
X_train_std = np.std(X_train, axis=0, keepdims=True)

# 标准化输入加速学习（可选）
# X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)
# X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)

np.random.seed(42)
h = 8
learn_rate = 0.4  # 学习率
num_epochs = 2500  # 训练轮数，每轮遍历一次所有 mini-batch
mini_batch_size = 64  # 每个 mini-batch 的样本数
dropout_rate = 0.2  # Dropout率

model = KSNet.KSSequential(
    KSNet.KSLinear(input_dim=2, output_dim=h),
    KSNet.KSReLU(),
    # KSNet.KSDropout(dropout_rate),
    KSNet.KSLinear(h, 1)
)

logit_loss = KSNet.KSBinaryLogisticLoss()
optimizer = KSNet.KSSGDOptimizer(model.parameters(), lr=learn_rate)

step = 0  # 全局迭代步数，跨 epoch 累计
loss_history = []  # 每 50 步记录一次当前 batch 的损失
train_acc_history = []  # 每 50 步记录一次训练集准确率
test_acc_history = []  # 每 50 步记录一次测试集准确率

for epoch in range(num_epochs):
    # 每个 epoch 重新打乱并切分 mini-batch
    mini_batches = generate_mini_batches(X_train, Y_train, mini_batch_size, seed=epoch)

    for X_batch, Y_batch in mini_batches:
        # train_step 内部完成 train()、zero_grad、前向、损失、反向、step
        loss = model.train_step(
            x=X_batch,
            label=Y_batch,
            loss_fn=logit_loss,
            optimizer=optimizer,
        )

        # 与 04activation_comparison_with_minibatch.py 保持一致：每 50 步记录一次
        if step % 50 == 0:
            loss_history.append(loss)
            model.eval()
            train_acc = np.mean((KSNet.KSSigmoid.apply(model(X_train)) > 0.5) == Y_train)
            train_acc_history.append(train_acc)
            test_acc = np.mean((KSNet.KSSigmoid.apply(model(X_test)) > 0.5) == Y_test)
            test_acc_history.append(test_acc)

            if step % 1000 == 0:
                print(f"Step {step}, Loss: {loss:.6f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        step += 1

# ==================== 损失与准确率可视化 ====================
total_steps = step  # 最终迭代步数
iterations = list(range(0, total_steps, 50))

plt.figure(figsize=(10, 6))
plt.plot(iterations, loss_history, 'r-', linewidth=2, label='Loss')
plt.plot(iterations, train_acc_history, 'b-', linewidth=2, label='Train Accuracy')
plt.plot(iterations, test_acc_history, 'g-', linewidth=2, label='Test Accuracy')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Mini-batch Training Progress', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([0, total_steps])
plt.show()
