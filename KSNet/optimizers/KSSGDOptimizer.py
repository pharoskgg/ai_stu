import numpy as np
from KSNet.core.KSOptimizerBase import KSOptimizerBase

class SGDOptimizer(KSOptimizerBase):
    """基础SGD梯度下降，带L2权重衰减"""
    def step(self) -> None:
        for param, grad in self._get_trainable_params():
            if self.weight_decay:
                param -= self.lr * (grad + self.weight_decay * param)
            else:
                param -= self.lr * grad
