import numpy as np

from KSNet.core.KSOptimizerBase import KSOptimizerBase


class KSAdamOptimizer(KSOptimizerBase):
    """Adam 优化器。

    对每个参数分别维护梯度的一阶矩和二阶原点矩，并在训练初期进行
    偏差修正：

    ``m = beta1 * m + (1 - beta1) * grad``
    ``v = beta2 * v + (1 - beta2) * grad**2``
    ``param -= lr * m_hat / (sqrt(v_hat) + epsilon)``

    ``weight_decay`` 沿用本项目 SGD、Momentum 和 RMSProp 的约定：只对
    ``layer.weight`` 添加 L2 正则项，不衰减 bias。
    """

    def __init__(
        self,
        layers,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 必须在 [0, 1) 范围内，实际为 {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 必须在 [0, 1) 范围内，实际为 {beta2}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon 必须大于 0，实际为 {epsilon}")

        super().__init__(layers, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_count = 0
        # 每个参数都维护独立的一阶矩 m 和二阶原点矩 v。
        self.first_moment: dict[int, np.ndarray] = {}
        self.second_moment: dict[int, np.ndarray] = {}

    def step(self) -> None:
        self.step_count += 1
        beta1_correction = 1.0 - self.beta1 ** self.step_count
        beta2_correction = 1.0 - self.beta2 ** self.step_count

        for layer in self.layers:
            if not layer.trainable:
                continue

            for param, grad in layer.parameters():
                if grad is None:
                    continue

                # 与其他优化器保持一致：仅对权重施加 L2 正则化。
                update_grad = grad
                if self.weight_decay and param is layer.weight:
                    update_grad = grad + self.weight_decay * param

                parameter_id = id(param)
                first_moment = self.first_moment.setdefault(
                    parameter_id, np.zeros_like(param)
                )
                second_moment = self.second_moment.setdefault(
                    parameter_id, np.zeros_like(param)
                )

                first_moment = (
                    self.beta1 * first_moment + (1.0 - self.beta1) * update_grad
                )
                second_moment = (
                    self.beta2 * second_moment
                    + (1.0 - self.beta2) * np.square(update_grad)
                )

                # 从 0 初始化的 EMA 在前几步会偏小，Adam 用偏差修正抵消它。
                corrected_first_moment = first_moment / beta1_correction
                corrected_second_moment = second_moment / beta2_correction
                param -= self.lr * corrected_first_moment / (
                    np.sqrt(corrected_second_moment) + self.epsilon
                )

                self.first_moment[parameter_id] = first_moment
                self.second_moment[parameter_id] = second_moment
