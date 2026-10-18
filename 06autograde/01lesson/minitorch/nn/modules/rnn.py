"""通过 Tensor 算子实现单层单向 RNN，接口参考 PyTorch。"""

from numbers import Integral

import numpy as np

from ...tensor import Tensor, stack
from .. import functional
from ..parameter import Parameter
from .module import Module


class RNN(Module):
    """单层单向 Elman RNN，返回 (output, h_n)。

    默认输入为 (seq_len, batch, input_size)，batch_first=True 时为
    (batch, seq_len, input_size)。无批次输入为 (seq_len, input_size)。
    隐藏状态保留层维度：有批次为 (1, batch, hidden_size)，无批次为
    (1, hidden_size)。支持 tanh/relu；当前 num_layers 只支持 1。

    h_t = activation(x_t @ weight_ih_l0.T + h_prev @ weight_hh_l0.T
                     + bias_ih_l0 + bias_hh_l0)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = "tanh", bias: bool = True,
                 batch_first: bool = False):
        super().__init__()
        for name, value in (
            ("input_size", input_size),
            ("hidden_size", hidden_size),
            ("num_layers", num_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} 必须是整数")
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")

        if num_layers != 1:
            raise NotImplementedError("RNN 当前只支持 num_layers=1（单层单向）")
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity 必须为 'tanh' 或 'relu'")
        if not isinstance(bias, bool) or not isinstance(batch_first, bool):
            raise TypeError("bias 和 batch_first 必须是 bool")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = False
        self.weight_ih_l0 = Parameter(np.empty((self.hidden_size, self.input_size)))
        self.weight_hh_l0 = Parameter(np.empty((self.hidden_size, self.hidden_size)))
        if bias:
            self.bias_ih_l0 = Parameter(np.empty(self.hidden_size))
            self.bias_hh_l0 = Parameter(np.empty(self.hidden_size))
        else:
            self.register_parameter("bias_ih_l0", None)
            self.register_parameter("bias_hh_l0", None)
        self.reset_parameters()

    def reset_parameters(self):
        """所有权重和偏置均匀初始化到 ±1/sqrt(hidden_size)。"""
        bound = 1 / np.sqrt(self.hidden_size)
        for parameter in self.parameters():
            parameter.data[:] = np.random.uniform(-bound, bound, parameter.shape)

    def forward(self, input: Tensor, hx: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if not isinstance(input, Tensor):
            raise TypeError("input 必须是 Tensor")
        if input.ndim not in (2, 3):
            raise ValueError("RNN input 必须是二维（无批次）或三维 Tensor")
        if input.shape[-1] != self.input_size:
            raise ValueError(f"输入特征数必须为 {self.input_size}，实际为 {input.shape[-1]}")

        is_batched = input.ndim == 3
        if not is_batched:
            sequence = input.reshape(input.shape[0], 1, self.input_size)
        else:
            sequence = input.transpose(0, 1) if self.batch_first else input
        seq_len, batch_size, _ = sequence.shape
        if seq_len == 0:
            raise ValueError("RNN 输入序列长度必须大于 0")

        expected_shape = ((1, batch_size, self.hidden_size) if is_batched
                          else (1, self.hidden_size))
        if hx is None:
            h = Tensor(np.zeros((batch_size, self.hidden_size)))
        else:
            if not isinstance(hx, Tensor):
                raise TypeError("hx 必须是 Tensor")
            if hx.shape != expected_shape:
                raise ValueError(
                    f"hx 的形状必须为 {expected_shape}，实际为 {hx.shape}"
                )
            h = hx.reshape(batch_size, self.hidden_size)

        weight_ih = self.weight_ih_l0.transpose(0, 1)
        weight_hh = self.weight_hh_l0.transpose(0, 1)
        activation = functional.tanh if self.nonlinearity == "tanh" else functional.relu
        outputs = []
        for t in range(seq_len):
            h = sequence.select(0, t) @ weight_ih + h @ weight_hh
            if self.bias:
                h = h + self.bias_ih_l0 + self.bias_hh_l0
            h = activation(h)
            outputs.append(h)

        output = stack(outputs)
        if not is_batched:
            return output.reshape(seq_len, self.hidden_size), h.reshape(1, self.hidden_size)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h.reshape(1, batch_size, self.hidden_size)
