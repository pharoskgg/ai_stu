from numbers import Real

import numpy as np

from KSNet.core.KSangNet import KSNet


class KSSoftmax(KSNet):
    """无参数 Softmax 激活层。

    输入和输出均为二维张量，形状为 ``(batch_size, dim)``。Softmax
    沿特征维度计算，每一行的输出之和为 1。

    :param epsilon: 归一化分母的下限，用于防止除零。数值稳定实现中分母
        至少为 1，因此该参数主要用于保持接口兼容。
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self._validate_epsilon(epsilon)
        self.trainable = False
        self.epsilon = float(epsilon)

    @staticmethod
    def _validate_epsilon(epsilon: float) -> None:
        if not isinstance(epsilon, Real) or isinstance(epsilon, (bool, np.bool_)):
            raise ValueError(f"epsilon 必须是有限实数，实际为 {epsilon!r}")
        if not np.isfinite(epsilon):
            raise ValueError(f"epsilon 必须是有限标量，实际为 {epsilon!r}")
        if epsilon <= 0 or epsilon > 1:
            raise ValueError(f"epsilon 必须在 (0, 1] 范围内，实际为 {epsilon!r}")

    @staticmethod
    def apply(x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """逐行计算数值稳定的 Softmax。"""
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Softmax 输入必须是 numpy.ndarray，实际为 {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(
                f"Softmax 输入必须为2维张量(batch, dim)，当前形状为 {x.shape}"
            )
        if x.shape[1] == 0:
            raise ValueError("Softmax 的特征维度不能为0")
        if not np.issubdtype(x.dtype, np.number) or np.issubdtype(
            x.dtype, np.complexfloating
        ):
            raise TypeError(f"Softmax 输入必须是实数数组，实际 dtype 为 {x.dtype}")
        if not np.all(np.isfinite(x)):
            raise ValueError("Softmax 输入不能包含 NaN 或无穷大")
        KSSoftmax._validate_epsilon(epsilon)

        # 每行先减去最大值，使最大的指数项恒为 1，避免 exp 溢出。
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        denominator = np.sum(exp_x, axis=1, keepdims=True)

        # 对有限、非空输入，denominator >= 1；maximum 是额外的除零保护，
        # 不会像直接加 epsilon 那样破坏“每行概率之和为 1”的性质。
        denominator = np.maximum(denominator, epsilon)
        return exp_x / denominator

    def forward(self, x: np.ndarray) -> np.ndarray:
        """计算 Softmax 输出并缓存反向传播所需的数据。"""
        output = self.apply(x, self.epsilon)
        self.input = x
        self.output = output
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """根据上游梯度 ``dout`` 计算损失对输入的梯度。"""
        if self.input is None or self.output is None:
            raise RuntimeError("请先执行 forward 再调用 backward")
        if not isinstance(dout, np.ndarray):
            raise TypeError(f"上游梯度必须是 numpy.ndarray，实际为 {type(dout).__name__}")
        if dout.shape != self.output.shape:
            raise ValueError(
                f"上游梯度形状不匹配，期望 {self.output.shape}，实际 {dout.shape}"
            )
        if not np.issubdtype(dout.dtype, np.number) or np.issubdtype(
            dout.dtype, np.complexfloating
        ):
            raise TypeError(f"上游梯度必须是实数数组，实际 dtype 为 {dout.dtype}")

        # J = diag(y) - y y^T。利用矩阵结构直接计算 J @ dout，
        # 避免为每个样本构造 dim x dim 的雅可比矩阵。
        dot = np.sum(dout * self.output, axis=1, keepdims=True)
        return self.output * (dout - dot)
