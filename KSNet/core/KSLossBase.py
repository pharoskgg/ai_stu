from abc import ABC, abstractmethod
import numpy as np

class KSLossBase(ABC):
    """损失函数统一抽象基类"""
    def __init__(self):
        # 缓存前向传播数据，反向求梯度使用
        self.pred: np.ndarray | None = None
        self.label: np.ndarray | None = None
        self.batch_size: int | None = None

    @abstractmethod
    def forward(self, pred: np.ndarray, label: np.ndarray) -> float:
        """
        前向：计算整体损失标量
        :param pred: 网络原始输出 (batch, 1) 未经过sigmoid
        :param label: 真实标签 0/1，shape 和 pred 一致
        :return: 平均 loss 标量
        """
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        """
        反向：返回 dL/dpred，传给最后一层 Linear 的 dout
        return shape 和 self.pred 完全一致 (batch, 1)
        """
        pass