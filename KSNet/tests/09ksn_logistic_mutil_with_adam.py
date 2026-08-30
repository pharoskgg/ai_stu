import sys
from pathlib import Path

# 把项目根目录 ai_stu 加进 sys.path，让 import KSNet 能找到包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import KSNet
import numpy as np
from KSNet.util import generate_moons_dataset, plot_training_history


(X_train, Y_train), (X_test, Y_test) = generate_moons_dataset(n_samples=600, noise=0.2)
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
learn_rate = 0.1  # 学习率
num_epochs = 20000  # 训练轮数，每轮遍历一次所有 mini-batch
dropout_rate = 0.9  # Dropout率
beta1 = 0.9 # 动量系数
beta2 = 0.999 # RMSProp系数

model = KSNet.KSSequential(
    KSNet.KSLinear(input_dim=2, output_dim=h),
    KSNet.KSReLU(),
    # KSNet.KSDropout(dropout_rate),
    KSNet.KSLinear(h, 1)
)

logit_loss = KSNet.KSBinaryLogisticLoss()
optimizer = KSNet.KSAdamOptimizer(
    model.parameters(), lr=learn_rate, beta1=beta1, beta2=beta2
)

step = 0  # 全局迭代步数，跨 epoch 累计
loss_history = []  # 每 100 步记录一次当前 batch 的损失
train_acc_history = []  # 每 100 步记录一次训练集准确率
test_acc_history = []  # 每 100 步记录一次测试集准确率

for epoch in range(num_epochs):

    # train_step 内部完成 train()、zero_grad、前向、损失、反向、step
    loss = model.train_step(
        x=X_train,
        label=Y_train,
        loss_fn=logit_loss,
        optimizer=optimizer,
    )

    # 与 04activation_comparison_with_minibatch.py 保持一致：每 50 步记录一次
    if step % 100 == 0:
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
# total_steps = step  # 最终迭代步数
# iterations = list(range(0, total_steps, 100))

# plot_training_history(
#     iterations,
#     loss_history,
#     train_acc_history,
#     test_acc_history,
#     title="Momentum Training Progress",
# )
