import numpy as np

from KSNet.core.KSOptimizerBase import KSOptimizerBase


class KSMomentumOptimizer(KSOptimizerBase):
    """带动量的 SGD 优化器。

    ``weight_decay`` 作用于传给优化器的全部参数。更新规则使用梯度的
    指数移动平均：

    ``velocity = beta * velocity + (1 - beta) * gradient``
    ``param -= lr * velocity``
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        beta: float = 0.9,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta 必须在 [0, 1) 范围内，实际为 {beta}")

        super().__init__(params, lr, weight_decay)
        self.beta = beta
        # 每个参数各自维护一个速度缓存，支持多层网络和 weight / bias 参数。
        self.velocity: dict[int, np.ndarray] = {}

    def step(self) -> None:
        for param, grad in self._get_trainable_params():
            update_grad = grad
            if self.weight_decay:
                update_grad = grad + self.weight_decay * param

            velocity = self.velocity.setdefault(id(param), np.zeros_like(param))

            velocity = self.beta * velocity + (1 - self.beta) * update_grad
            # 不能写成param = param - self.lr * velocity，因为 param 是局部变量，无法修改原始参数。
            # param -= self.lr * velocity 等价于 param.__isub__(self.lr * velocity)，会修改原始参数。
            param -= self.lr * velocity
            self.velocity[id(param)] = velocity  # 更新速度缓存
