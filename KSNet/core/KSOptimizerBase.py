from abc import ABC, abstractmethod
from collections.abc import Iterable
from KSNet.core.KSangNet import KSNet

class KSOptimizerBase(ABC):
    """优化器统一抽象基类"""
    def __init__(self, layers: Iterable[KSNet], lr: float, weight_decay: float = 0.0):
        self.layers = list(layers) # 了让优化器能直接接收 model，否则调用传參为:model.layers
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        """批量清空所有层梯度，通用逻辑放入基类，子类无需重复实现"""
        for layer in self.layers:
            layer.zero_grad()

    @abstractmethod
    def step(self) -> None:
        """参数更新核心逻辑，各优化器自行实现"""
        pass

    def _get_trainable_params(self):
        """内置工具：遍历所有层，只返回可训练的(参数,梯度)对"""
        for layer in self.layers:
            if not layer.trainable:
                continue
            for param, grad in layer.parameters():
                if grad is None:
                    continue
                yield param, grad
