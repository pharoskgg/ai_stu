"""使用项目内 minitorch 构建多层全连接网络，完成月牙数据二分类。

参考 KSNet/tests/02ksn_logistic_mutil.py 的实验流程。
依赖 numpy、scikit-learn、matplotlib；在项目根目录运行：
    python 06autograde/01lesson/experiment/01fnn/01fnn_binary_classifier.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 导入本课实现的 minitorch，不受启动脚本时的工作目录影响。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from minitorch import Tensor, nn, optim


class BinaryClassifier(nn.Module):
    """两个隐藏层的 FNN，输出类别 0、1 的两个原始分数（logits）。"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.network(x)


def plot_results(model, x, y, split, steps, losses):
    """绘制训练损失、预测区域和真实样本，红圈标记测试集错分点。"""
    model.eval()
    figure, (loss_ax, prediction_ax) = plt.subplots(1, 2, figsize=(13, 5))
    loss_ax.plot(steps, losses, color="tab:red", label="Training loss")
    loss_ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training Progress")
    loss_ax.grid(alpha=0.3)
    loss_ax.legend()

    # 用 minitorch 对平面网格前向预测，得到分类区域和决策边界。
    lower = x.min(axis=0) - 0.5
    upper = x.max(axis=0) + 0.5
    xx, yy = np.meshgrid(
        np.arange(lower[0], upper[0], 0.02),
        np.arange(lower[1], upper[1], 0.02),
    )
    grid = Tensor(np.c_[xx.ravel(), yy.ravel()])
    regions = model(grid).data.argmax(axis=1).reshape(xx.shape)
    prediction_ax.contourf(
        xx, yy, regions, levels=[-0.5, 0.5, 1.5], cmap="RdBu", alpha=0.25
    )
    for data, labels, marker, name in (
        (x[:split], y[:split], "o", "Train (true labels)"),
        (x[split:], y[split:], "^", "Test (true labels)"),
    ):
        prediction_ax.scatter(
            data[:, 0], data[:, 1], c=labels, cmap="RdBu", vmin=0, vmax=1,
            marker=marker, edgecolors="k", linewidths=0.5, s=35, label=name,
        )

    test_predictions = model(Tensor(x[split:])).data.argmax(axis=1)
    wrong = x[split:][test_predictions != y[split:]]
    if len(wrong):
        prediction_ax.scatter(
            wrong[:, 0], wrong[:, 1], facecolors="none", edgecolors="red",
            linewidths=1.5, s=130, label="Misclassified test samples",
        )
    accuracy = np.mean(test_predictions == y[split:])
    prediction_ax.set(
        xlabel="Feature 1", ylabel="Feature 2",
        title=f"Decision Boundary (test accuracy: {accuracy:.2%})",
        xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
    )
    prediction_ax.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    # 先保存再显示，避免关闭窗口后保存出空白图片。
    output_path = Path(__file__).with_name("fnn_binary_predictions.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"预测图已保存：{output_path}")
    plt.show()


def main():
    seed = 42
    epochs = 3000
    learning_rate = 0.1
    log_interval = 500

    np.random.seed(seed)
    x, y = make_moons(n_samples=300, noise=0.2, random_state=seed)
    # 与参考脚本相同，前 80% 为训练集，其余为测试集。
    split = int(len(x) * 0.8)
    x_train, x_test = Tensor(x[:split]), Tensor(x[split:])
    # CrossEntropyLoss 接受形状 (N,) 的类别索引，不需要 one-hot 或列向量。
    y_train, y_test = Tensor(y[:split]), Tensor(y[split:])

    model = BinaryClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    steps, losses = [], []

    print("minitorch 全连接二分类网络：2 → 16 → 8 → 2")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(x_train)
        # 交叉熵内部完成 log-softmax，直接传入原始 logits。
        loss = criterion(logits, y_train)
        loss.backward()  # minitorch 自动完成所有层的反向传播。
        optimizer.step()

        if epoch == 1 or epoch % log_interval == 0:
            steps.append(epoch)
            losses.append(float(loss.data))
            print(f"Epoch {epoch:4d}/{epochs}, Loss: {float(loss.data):.6f}")

    model.eval()
    for name, inputs, targets in (
        ("训练集", x_train, y_train),
        ("测试集", x_test, y_test),
    ):
        logits = model(inputs)
        loss = float(criterion(logits, targets).data)
        # 两个类别分数取最大者；计算准确率无需额外执行 softmax。
        predictions = logits.data.argmax(axis=1)
        accuracy = np.mean(predictions == targets.data)
        print(f"{name}：Loss = {loss:.6f}, 准确率 = {accuracy:.2%}")

    plot_results(model, x, y, split, steps, losses)


if __name__ == "__main__":
    main()
