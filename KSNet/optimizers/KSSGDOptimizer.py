from KSNet.core.KSOptimizerBase import KSOptimizerBase

class SGDOptimizer(KSOptimizerBase):
    """基础 SGD；weight_decay 作用于传给优化器的全部参数。"""

    def step(self) -> None:
        for param, grad in self._get_trainable_params():
            update_grad = grad
            if self.weight_decay:
                update_grad = grad + self.weight_decay * param
            param -= self.lr * update_grad
