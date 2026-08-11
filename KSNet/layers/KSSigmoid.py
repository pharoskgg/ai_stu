from KSNet.core.KSangNet import KSNet
import numpy as np

class KSSigmoid(KSNet):
    """无参数 Sigmoid 激活层：y = 1 / (1 + exp(-x))。"""
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播，计算输出
        :param x: 输入张量，形状 (batch_size, dim)
        :return: 输出张量，形状 (batch_size, dim)
        """
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.input is None or self.output is None:
            raise RuntimeError("请先执行 forward 再调用 backward")

        if dout.shape != self.output.shape:
            raise ValueError(
                f"上游梯度形状不匹配，期望 {self.output.shape}，实际 {dout.shape}"
            )

        # Sigmoid的导数: a * (1 - a)
        return dout * self.output * (1 - self.output)