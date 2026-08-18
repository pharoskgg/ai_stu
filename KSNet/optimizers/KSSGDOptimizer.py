import numpy as np
from KSNet.core.KSOptimizerBase import KSOptimizerBase

class SGDOptimizer(KSOptimizerBase):
    """基础 SGD，与 PyTorch 一致按 grad + weight_decay * weight 更新，偏置不衰减。"""
    def step(self) -> None:
        for layer in self.layers:
            if not layer.trainable:
                continue

            for param, grad in layer.parameters():
                if grad is None:
                    continue

                # parameters() 返回的是单个参数及其梯度，而不是“权重、偏置”数组。
                # 只有当前参数确实是该层权重时才应用权重衰减，偏置仅按梯度更新。
                if self.weight_decay and param is layer.weight:
                    param -= self.lr * (grad + self.weight_decay * param)
                else:
                    param -= self.lr * grad
