import platform
from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def setup_chinese_font():
    """
    设置 matplotlib 中文字体，兼容 macOS / Windows / Linux
    """
    system_name = platform.system()
    if system_name == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'STHeiti']
    elif system_name == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    # 解决坐标轴负号显示问题
    plt.rcParams['axes.unicode_minus'] = False


def plot_training_history(
    steps: Sequence[int],
    losses: Sequence[float],
    train_accuracies: Optional[Sequence[float]] = None,
    test_accuracies: Optional[Sequence[float]] = None,
    *,
    title: str = "Training Progress",
    show: bool = True,
) -> tuple[Figure, Axes]:
    """绘制训练过程中的损失和可选的训练/测试准确率曲线。

    所有传入的历史记录必须与 ``steps`` 等长。返回 Figure 和 Axes，
    以便调用方继续添加标注、保存图片或调整样式。
    """
    if len(steps) == 0:
        raise ValueError("steps 不能为空")
    if len(losses) != len(steps):
        raise ValueError("losses 与 steps 的长度必须一致")
    if train_accuracies is not None and len(train_accuracies) != len(steps):
        raise ValueError("train_accuracies 与 steps 的长度必须一致")
    if test_accuracies is not None and len(test_accuracies) != len(steps):
        raise ValueError("test_accuracies 与 steps 的长度必须一致")

    figure, axes = plt.subplots(figsize=(10, 6))
    axes.plot(steps, losses, "r-", linewidth=2, label="Loss")
    if train_accuracies is not None:
        axes.plot(steps, train_accuracies, "b-", linewidth=2, label="Train Accuracy")
    if test_accuracies is not None:
        axes.plot(steps, test_accuracies, "g-", linewidth=2, label="Test Accuracy")

    axes.set_xlabel("Step", fontsize=12)
    axes.set_ylabel("Value", fontsize=12)
    axes.set_title(title, fontsize=14, fontweight="bold")
    axes.legend(loc="best", fontsize=10)
    axes.grid(True, alpha=0.3)
    if len(steps) > 1:
        axes.set_xlim(steps[0], steps[-1])

    if show:
        plt.show()

    return figure, axes
