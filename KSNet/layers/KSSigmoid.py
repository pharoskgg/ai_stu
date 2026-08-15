from KSNet.core.KSangNet import KSNet
import numpy as np

class KSSigmoid(KSNet):
    """无参数 Sigmoid 激活层，使用数值稳定的公式计算。"""
    def __init__(self):
        super().__init__()
        self.trainable = False

    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        # 根据输入正负选择等价公式，避免 np.exp() 的输入过大而溢出：
        # x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
        # x <  0: sigmoid(x) = exp(x) / (1 + exp(x))
        mask = x >= 0
        output = np.empty_like(x, dtype=np.float64)
        output[mask] = 1.0 / (1.0 + np.exp(-x[mask]))

        exp_x = np.exp(x[~mask])
        output[~mask] = exp_x / (1.0 + exp_x)
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播，计算输出
        :param x: 输入张量，形状 (batch_size, dim)
        :return: 输出张量，形状 (batch_size, dim)
        """
        self.input = x
        self.output = self.apply(x)
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
