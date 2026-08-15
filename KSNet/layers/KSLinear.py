from KSNet.core.KSangNet import KSNet
import numpy as np

class KSLinear(KSNet):
    """
    线性层（全连接层）实现
    输入输出都是二维张量，形状 (batch_size, dim)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # 参数合法性校验
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"维度必须大于0，input_dim={input_dim}, output_dim={output_dim}")
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Xavier初始化，替代单纯randn*0.01，提升数值稳定性
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weight = np.random.randn(input_dim, output_dim) * scale
        self.bias = np.zeros(output_dim)

        # 预分配梯度数组，zero_grad可原地清零
        self.w_grad = np.zeros_like(self.weight)
        self.b_grad = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播，计算输出
        :param x: 输入张量，形状 (batch_size, input_dim)
        :return: 输出张量，形状 (batch_size, output_dim)
        """
        # 输入维度校验
        if x.ndim != 2:
            raise ValueError(f"Linear输入必须为2维张量(batch, dim)，当前维度{x.shape}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"输入特征维度不匹配，期望{self.input_dim}，实际{x.shape[-1]}")

        self.input = x
        out = np.dot(x, self.weight) + self.bias
        self.output = out 
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播，计算梯度
        :param dout: 上游梯度，形状 (batch_size, output_dim)
        :return: 下游梯度，形状 (batch_size, input_dim)
        """
        if self.input is None:
            raise RuntimeError("请先执行forward再调用backward")
        if dout.shape != self.output.shape:
            raise ValueError(f"上游梯度形状不匹配，期望{self.output.shape}，实际{dout.shape}")

        # 冻结参数时跳过权重梯度计算，节省算力
        if self.trainable:
            self.w_grad += np.dot(self.input.T, dout)
            self.b_grad += np.sum(dout, axis=0)

        # 输入梯度必须计算，传递给上一层
        dx = np.dot(dout, self.weight.T)
        return dx