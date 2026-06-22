from KSNet.KSangNet import KSangNet
import numpy as np

class KSLinear(KSangNet):
    """
    线性层（全连接层）实现
    输入输出都是二维张量，形状 (batch_size, dim)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # 初始化权重和偏置参数
        self.weight = np.random.randn(input_dim, output_dim) * 0.01  # 小随机数初始化
        self.bias = np.zeros(output_dim)  # 偏置初始化为零

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播，计算输出
        :param x: 输入张量，形状 (batch_size, input_dim)
        :return: 输出张量，形状 (batch_size, output_dim)
        """
        self.input = x  # 保存输入张量，用于反向传播时使用
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播，计算梯度
        :param dout: 上游梯度，形状 (batch_size, output_dim)
        :return: 下游梯度，形状 (batch_size, input_dim)
        """
        # 计算权重梯度
        self.w_grad = np.dot(self.input.T, dout)  # (input_dim, batch_size) @ (batch_size, output_dim) -> (input_dim, output_dim)
        # 计算偏置梯度
        self.b_grad = np.sum(dout, axis=0)  # (output_dim,)
        # 计算输入梯度，传递给上一层
        dx = np.dot(dout, self.weight.T)  # (batch_size, output_dim) @ (output_dim, input_dim) -> (batch_size, input_dim)
        return dx