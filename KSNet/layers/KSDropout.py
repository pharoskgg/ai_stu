from KSNet.core.KSangNet import KSNet
import numpy as np

class KSDropout(KSNet):
    """
    Dropout层实现
    输入输出都是二维张量，形状 (batch_size, dim)
    """
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        if not (0 <= dropout_rate < 1):
            raise ValueError(f"dropout_rate必须在[0, 1)范围内，当前值为{dropout_rate}")
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if self.training:
            self.mask = (np.random.rand(*x.shape) >= self.dropout_rate).astype(float)
            out = x * self.mask / (1 - self.dropout_rate)  # 缩放以保持期望值不变
        else:
            out = x  # 测试阶段不进行dropout
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("请先执行forward再调用backward")
        dx = dout * self.mask / (1 - self.dropout_rate)  # 缩放以保持期望值不变
        return dx