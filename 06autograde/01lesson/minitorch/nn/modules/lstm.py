from numbers import Integral

import numpy as np

from ...tensor import Tensor, stack
from .. import functional
from ..parameter import Parameter
from .module import Module


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        for name, value in (("input_size", input_size), ("hidden_size", hidden_size)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} 必须是整数")
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cache = None  # 用于保存前向传播中间结果，便于反向传播计算梯度

        # 循环神经网络是按时间步来处理的，所以可以先不管一句话的长度，先把每个时间步的输入和隐藏状态的维度确定下来。
        # input: (batch_size, input_size) * weight: (input_size, hidden_size) = (batch_size, hidden_size)
        # hidden: (batch_size, hidden_size)
        self.weight_input = Parameter(np.empty((input_size, hidden_size)))
        self.hidden_input = Parameter(np.empty((hidden_size, hidden_size)))
        self.b_input = Parameter(np.empty((hidden_size)))

        self.weight_forget = Parameter(np.empty((input_size, hidden_size)))
        self.hidden_forget = Parameter(np.empty((hidden_size, hidden_size)))
        self.b_forget = Parameter(np.empty((hidden_size)))

        self.weight_output = Parameter(np.empty((input_size, hidden_size)))
        self.hidden_output = Parameter(np.empty((hidden_size, hidden_size)))
        self.b_output = Parameter(np.empty((hidden_size)))

        self.weight_cell = Parameter(np.empty((input_size, hidden_size)))
        self.hidden_cell = Parameter(np.empty((hidden_size, hidden_size)))
        self.b_cell = Parameter(np.empty((hidden_size)))

        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重和偏置。"""
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data[:] = np.random.uniform(-stdv, stdv, size=weight.shape)

    def forward(self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # input: (seq_len, batch_size, input_size)
        # hx: (h_0, c_0), h_0: (batch_size, hidden_size), c_0: (batch_size, hidden_size)
        # output: (seq_len, batch_size, hidden_size), (h_n, c_n), h_n:

        # 入参形状检查
        if not isinstance(input, Tensor):
            raise TypeError("input 必须是 Tensor")
        if input.ndim != 3:
            raise ValueError("input 必须为 (seq_len, batch_size, input_size)")
        if input.shape[2] != self.input_size:
            raise ValueError(f"输入特征数必须为 {self.input_size}")
        if input.shape[0] == 0:
            raise ValueError("序列长度必须大于 0")
        
        batch_size = input.shape[1]
        state_shape = (1, batch_size, self.hidden_size)
        
        if hx is None:
            h = Tensor(np.zeros((input.shape[1], self.hidden_size)))
            c_t = Tensor(np.zeros((input.shape[1], self.hidden_size)))
        else:
            if not isinstance(hx, tuple) or len(hx) != 2:
                raise TypeError("hx 必须为 (h0, c0)")

            h0, c0 = hx
            if not isinstance(h0, Tensor) or not isinstance(c0, Tensor):
                raise TypeError("h0 和 c0 必须是 Tensor")
            if h0.shape != state_shape or c0.shape != state_shape:
                raise ValueError(f"h0 和 c0 的形状必须为 {state_shape}")

            # 对外保留层维度；单层内部计算使用 (B, H)
            h = h0.select(0, 0)
            c_t = c0.select(0, 0)


        self.cache = []
        for t in range(input.shape[0]):
            # (input_size) @ (input_size, hidden_size) + (hidden_size) @ (hidden_size, hidden_size) + (hidden_size)
            # = (hidden_size)
            x = input.select(0, t)  # 取出当前时间步的输入，形状为 (batch_size, input_size)
            input_gate = functional.sigmoid(
                x @ self.weight_input + h @ self.hidden_input + self.b_input
            )

            forget_gate = functional.sigmoid(
                x @ self.weight_forget + h @ self.hidden_forget + self.b_forget
            )

            output_gate = functional.sigmoid(
                x @ self.weight_output + h @ self.hidden_output + self.b_output
            )

            cell_candidate = functional.tanh(
                x @ self.weight_cell + h @ self.hidden_cell + self.b_cell
            )

            c_t = forget_gate * c_t + input_gate * cell_candidate
            h = output_gate * functional.tanh(c_t)

            self.cache.append((input_gate, forget_gate, output_gate, cell_candidate, c_t, h))

        output = stack([step[-1] for step in self.cache])
        h_n = h.reshape(1, batch_size, self.hidden_size)
        c_n = c_t.reshape(1, batch_size, self.hidden_size)

        return output, (h_n, c_n)