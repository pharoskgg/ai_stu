"""普通随机梯度下降：直接更新叶子 Tensor 的数据。"""

import math
from numbers import Real

from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        if isinstance(lr, bool) or not isinstance(lr, Real):
            raise TypeError("learning rate must be a real number")
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("learning rate must be finite and non-negative")
        super().__init__(params)
        self.lr = lr

    def step(self):
        for parameter in self.params:
            if parameter.grad is not None:
                parameter.data -= self.lr * parameter.grad
