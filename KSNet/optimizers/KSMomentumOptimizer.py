import numpy as np

from KSNet.core.KSOptimizerBase import KSOptimizerBase


class KSMomentumOptimizer(KSOptimizerBase):
    """带动量的 SGD 优化器。

    ``weight_decay`` 与本项目的 ``SGDOptimizer`` 保持一致：只作用于
    ``layer.weight``，不作用于偏置。更新规则与 PyTorch SGD 的默认
    动量是梯度的指数移动平均：

    ``velocity = beta * velocity + (1 - beta) * gradient``
    ``param -= lr * velocity``
    """

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
        # 每个参数各自维护一个速度缓存，支持多层网络和 weight / bias 参数。
        self.velocity: dict[int, np.ndarray] = {}

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

                velocity = self.velocity.setdefault(id(param), np.zeros_like(param))

                velocity = self.beta * velocity + (1 - self.beta) * update_grad
                # 不能写成param = param - self.lr * velocity，因为 param 是局部变量，无法修改原始参数。
                # param -= self.lr * velocity 等价于 param.__isub__(self.lr * velocity)，会修改原始参数。
                param -= self.lr * velocity
                self.velocity[id(param)] = velocity  # 更新速度缓存
