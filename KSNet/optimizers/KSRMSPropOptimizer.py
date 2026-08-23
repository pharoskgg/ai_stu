import numpy as np

from KSNet.core.KSOptimizerBase import KSOptimizerBase


class KSRMSPropOptimizer(KSOptimizerBase):
    def __init__(
        self,
        layers,
        lr: float = 0.01,
        beta: float = 0.9,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta 必须在 [0, 1) 范围内，实际为 {beta}")

        super().__init__(layers, lr, weight_decay)
        self.beta = beta
        # 每个参数各自维护一个平方梯度平均值缓存，支持多层网络和 weight / bias 参数。
        self.squared_grad_avg: dict[int, np.ndarray] = {}

    def step(self) -> None:
        for layer in self.layers:
            if not layer.trainable:
                continue

            for param, grad in layer.parameters():
                if grad is None:
                    continue

                # 与 SGDOptimizer 的约定一致：仅对权重施加 L2 正则化。
                update_grad = grad
                if self.weight_decay and param is layer.weight:
                    update_grad = grad + self.weight_decay * param

                squared_grad_avg = self.squared_grad_avg.setdefault(id(param), np.zeros_like(param))

                squared_grad_avg = self.beta * squared_grad_avg + (1 - self.beta) * np.square(update_grad)

                param -= self.lr * (update_grad / (np.sqrt(squared_grad_avg) + 1e-8))
                # 不能写成param = param - self.lr * squared_grad_avg，因为 param 是局部变量，无法修改原始参数。
                # param -= self.lr * squared_grad_avg 等价于 param.__isub__(self.lr * squared_grad_avg)，会修改原始参数。
                self.squared_grad_avg[id(param)] = squared_grad_avg  # 更新平方梯度平均值缓存


