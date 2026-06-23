import numpy as np
from KSNet.core.KSOptimizerBase import KSOptimizerBase

class SGDOptimizer(KSOptimizerBase):
    """基础SGD梯度下降，带L2权重衰减"""
    def step(self) -> None:
        for param, grad in self._get_trainable_params():
            # L2正则：loss += 0.5 * wd * w^2 → 梯度 += wd * w
            # grad_with_decay = grad + self.weight_decay * param
            # 梯度下降更新
            param -= self.lr * grad