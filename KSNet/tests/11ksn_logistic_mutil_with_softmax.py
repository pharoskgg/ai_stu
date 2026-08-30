import sys
from pathlib import Path

# 把项目根目录 ai_stu 加进 sys.path，让 import KSNet 能找到包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import KSNet
from KSNet.util import generate_multiclass_dataset


# ==================== 1. 预测与准确率 ====================
def predict(model, X):
    """返回类别索引；Softmax 概率最大的列就是预测类别。"""
    probabilities = model.predict(X)
    return np.argmax(probabilities, axis=1)


def accuracy(y_pred, y_true_onehot):
    """将 One-hot 标签还原为类别索引后计算准确率。"""
    y_true = np.argmax(y_true_onehot, axis=1)
    return float(np.mean(y_pred == y_true))


# ==================== 2. KSNet 模型训练 ====================
def train_neural_network(
    X_train,
    Y_train,
    X_test,
    Y_test,
    *,
    num_epochs=40000,
    learn_rate=0.2,
    hidden_dim=4,
    log_every=1000,
):
    """使用全量 SGD 训练 ReLU + Softmax 多分类网络。"""
    if num_epochs <= 0:
        raise ValueError(f"num_epochs 必须大于0，实际为 {num_epochs}")
    if log_every <= 0:
        raise ValueError(f"log_every 必须大于0，实际为 {log_every}")

    np.random.seed(42)
    input_dim = X_train.shape[1]
    num_classes = Y_train.shape[1]

    # 与 09activation_comparison_softmax.py 保持一致：输出层先通过 Softmax
    # 得到每个类别的概率，再交给交叉熵计算损失。
    model = KSNet.KSSequential(
        KSNet.KSLinear(input_dim=input_dim, output_dim=hidden_dim),
        KSNet.KSReLU(),
        KSNet.KSLinear(input_dim=hidden_dim, output_dim=num_classes),
        KSNet.KSSoftmax(),
    )
    # from_logits=False 表示 forward 接收的是 Softmax 概率；反向时损失先返回
    # dL/dp，随后 KSSequential 会调用 KSSoftmax.backward() 得到 dL/dlogits。
    loss_fn = KSNet.KSSoftmaxCrossEntropyLoss(from_logits=False)
    optimizer = KSNet.KSSGDOptimizer(model.parameters(), lr=learn_rate)

    print(f"\n{'=' * 60}")
    print("训练 KSNet ReLU + Softmax 多分类模型")
    print(f"{'=' * 60}")

    for step in range(num_epochs):
        # train_step 依次完成：训练模式、梯度清零、前向、损失、反向、参数更新。
        loss = model.train_step(x=X_train, label=Y_train, loss_fn=loss_fn, optimizer=optimizer, )

        if step % log_every == 0:
            train_acc = accuracy(predict(model, X_train), Y_train)
            test_acc = accuracy(predict(model, X_test), Y_test)
            print(
                f"Step {step:5d}, Loss: {loss:.6f}, "
                f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
            )

    final_train_acc = accuracy(predict(model, X_train), Y_train)
    final_test_acc = accuracy(predict(model, X_test), Y_test)
    print(
        f"\n最终结果 - 训练集准确率: {final_train_acc * 100:.2f}%, "
        f"测试集准确率: {final_test_acc * 100:.2f}%"
    )

    return {
        "model": model,
        "final_loss": loss,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
    }


# ==================== 3. 主函数 ====================
def main(num_epochs=40000):
    (X_train, Y_train), (X_test, Y_test) = generate_multiclass_dataset()
    print(f"训练集: X{X_train.shape}, Y{Y_train.shape}")
    print(f"测试集: X{X_test.shape}, Y{Y_test.shape}")

    results = train_neural_network(
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_epochs=num_epochs,
        learn_rate=0.2,
        hidden_dim=4,
    )
    return results


if __name__ == "__main__":
    main()
