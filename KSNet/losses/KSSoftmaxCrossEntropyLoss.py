from __future__ import annotations

from numbers import Real

import numpy as np

from KSNet.core.KSLossBase import KSLossBase


class SoftmaxCrossEntropyLoss(KSLossBase):
    """多分类 Softmax 交叉熵损失。

    默认接收形状为 ``(batch_size, num_classes)`` 的原始 logits，并在内部
    使用数值稳定的 Softmax。标签既可以是形状 ``(batch_size,)`` 或
    ``(batch_size, 1)`` 的类别索引，也可以是与预测同形状的 one-hot 标签。

    如果模型末尾已经包含 ``KSSoftmax``，请设置
    ``from_logits=False``，此时输入应为概率，反向返回对概率的梯度，再由
    Softmax 层继续反向传播。

    :param from_logits: ``True`` 表示输入为 logits；``False`` 表示输入为概率。
    :param epsilon: 概率模式下 ``log`` 和除法使用的最小概率。
    """

    def __init__(self, from_logits: bool = True, epsilon: float = 1e-12):
        super().__init__()
        if not isinstance(from_logits, (bool, np.bool_)):
            raise TypeError("from_logits 必须是布尔值")
        if not isinstance(epsilon, Real) or isinstance(epsilon, (bool, np.bool_)):
            raise ValueError(f"epsilon 必须是有限实数，实际为 {epsilon!r}")
        if not np.isfinite(epsilon) or epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"epsilon 必须在 (0, 1) 范围内，实际为 {epsilon!r}")

        self.from_logits = bool(from_logits)
        self.epsilon = float(epsilon)
        self.softmax_out: np.ndarray | None = None
        self._target: np.ndarray | None = None

    @staticmethod
    def _validate_prediction(pred: np.ndarray) -> None:
        if not isinstance(pred, np.ndarray):
            raise TypeError(f"pred 必须是 numpy.ndarray，实际为 {type(pred).__name__}")
        if pred.ndim != 2:
            raise ValueError(f"pred 必须为2维张量(batch, classes)，当前形状为 {pred.shape}")
        if pred.shape[0] == 0:
            raise ValueError("pred 的 batch_size 不能为0")
        if pred.shape[1] < 2:
            raise ValueError("多分类损失至少需要2个类别")
        if not np.issubdtype(pred.dtype, np.number) or np.issubdtype(
            pred.dtype, np.complexfloating
        ):
            raise TypeError(f"pred 必须是实数数组，实际 dtype 为 {pred.dtype}")
        if not np.all(np.isfinite(pred)):
            raise ValueError("pred 不能包含 NaN 或无穷大")

    @staticmethod
    def _make_target(label: np.ndarray, pred_shape: tuple[int, int]) -> np.ndarray:
        """将类别索引或稠密标签统一整理成 ``(N, C)`` 的目标分布。"""
        if not isinstance(label, np.ndarray):
            raise TypeError(f"label 必须是 numpy.ndarray，实际为 {type(label).__name__}")

        batch_size, num_classes = pred_shape

        # 如果 label 和 pred 的形状相同，它已经是稠密目标分布：
        # one-hot 示例 [0, 1, 0]；软标签示例 [0.05, 0.9, 0.05]。
        # 这种情况只需校验，无须再次编码。
        if label.shape == pred_shape:
            if not np.issubdtype(label.dtype, np.number) or np.issubdtype(
                label.dtype, np.complexfloating
            ):
                raise TypeError(f"label 必须是实数数组，实际 dtype 为 {label.dtype}")
            if not np.all(np.isfinite(label)):
                raise ValueError("label 不能包含 NaN 或无穷大")
            if np.any(label < 0) or np.any(label > 1):
                raise ValueError("one-hot/软标签的值必须在 [0, 1] 范围内")
            if not np.allclose(np.sum(label, axis=1), 1.0, rtol=1e-6, atol=1e-8):
                raise ValueError("one-hot/软标签每一行的和必须为1")
            return label.astype(np.float64, copy=False)

        # 否则按稀疏类别索引处理。兼容数据集中常见的 (N,) 和 (N, 1)
        # 两种形状，并统一压成一维，方便后续索引。
        if label.shape == (batch_size, 1):
            sparse_label = label[:, 0]
        elif label.shape == (batch_size,):
            sparse_label = label
        else:
            raise ValueError(
                "label 形状必须为类别索引 (batch,) / (batch, 1)，或与 pred "
                f"相同的 one-hot 形状 {pred_shape}，实际为 {label.shape}"
            )

        if not np.issubdtype(sparse_label.dtype, np.integer):
            raise TypeError("类别索引 label 必须使用整数 dtype")
        if np.any(sparse_label < 0) or np.any(sparse_label >= num_classes):
            raise ValueError(f"类别索引必须在 [0, {num_classes - 1}] 范围内")

        # 用 NumPy 高级索引一次性完成 one-hot 编码：第 i 个样本的类别为
        # sparse_label[i]，因此将 target[i, sparse_label[i]] 设置为 1。
        # 例如 [0, 2, 1] 会转换成 [[1,0,0], [0,0,1], [0,1,0]]。
        target = np.zeros(pred_shape, dtype=np.float64)
        target[np.arange(batch_size), sparse_label.astype(np.intp, copy=False)] = 1.0
        return target

    def forward(self, pred: np.ndarray, label: np.ndarray) -> float:
        """计算一个 batch 的平均交叉熵损失。"""
        self._validate_prediction(pred)
        target = self._make_target(label, pred.shape)
        batch_size = pred.shape[0]

        if self.from_logits:
            # 融合模式直接计算 log-softmax：先减去每行最大值可避免 exp 溢出；
            # 不先算 Softmax 再取 log，则可以避免极小概率变成 0 后出现 log(0)。
            shifted = pred - np.max(pred, axis=1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
            log_probabilities = shifted - log_sum_exp
            probabilities = np.exp(log_probabilities)
        else:
            # 显式 Softmax 模式下 pred 已经是概率，只负责检查它是否合法。
            if np.any(pred < 0) or np.any(pred > 1):
                raise ValueError("from_logits=False 时 pred 的值必须在 [0, 1] 范围内")
            if not np.allclose(np.sum(pred, axis=1), 1.0, rtol=1e-6, atol=1e-8):
                raise ValueError("from_logits=False 时 pred 每一行的概率和必须为1")
            probabilities = pred
            log_probabilities = np.log(np.clip(pred, self.epsilon, 1.0))

        # 交叉熵：L_i = -sum(target_i * log(probability_i))。
        # one-hot 中只有真实类别位置为 1；软标签则会对多个类别加权。
        loss_per_sample = -np.sum(target * log_probabilities, axis=1)

        # 所有缓存只在输入校验及计算成功后更新。
        self.pred = pred
        self.label = label
        self.batch_size = batch_size
        self.softmax_out = probabilities
        self._target = target
        self.loss = loss_per_sample
        return float(np.mean(loss_per_sample))

    def backward(self) -> np.ndarray:
        """返回平均损失对 ``forward`` 输入的梯度。"""
        if (
            self.pred is None
            or self.softmax_out is None
            or self._target is None
            or self.batch_size is None
        ):
            raise RuntimeError("请先执行 forward 再调用 backward")

        if self.from_logits:
            # Softmax 和交叉熵联合求导后可化简为 (p - y) / N，既高效又稳定。
            gradient = (self.softmax_out - self._target) / self.batch_size
        else:
            # 此处只计算交叉熵对概率 p 的梯度 -y/(N*p)；返回后还需要经过
            # KSSoftmax.backward()，才能得到损失对 logits 的梯度。
            safe_probabilities = np.clip(self.softmax_out, self.epsilon, 1.0)
            gradient = -self._target / (safe_probabilities * self.batch_size)

        self.dz = gradient
        return gradient
