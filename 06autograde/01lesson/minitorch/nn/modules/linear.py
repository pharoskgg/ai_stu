"""通过现有 Tensor 算子组合实现全连接层。"""

from numbers import Integral

import numpy as np

from ...tensor import Tensor
from ..parameter import Parameter
from .module import Module


class Linear(Module):
    """输入形状 (..., in_features)，输出形状 (..., out_features)。

    输入至少二维；weight 的形状为 (in_features, out_features)。
    前向计算 x @ weight + bias，反向传播由现有自动求导完成。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        for name, value in (("in_features", in_features), ("out_features", out_features)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} 必须是整数")
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        bound = 1 / np.sqrt(self.in_features)
        self.weight = Parameter(np.random.uniform(
            -bound, bound, (self.in_features, self.out_features)
        ))
        if bias:
            self.bias = Parameter(np.random.uniform(-bound, bound, self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim < 2:
            raise ValueError("Linear 输入必须至少为二维，形状为 (..., in_features)")
        if x.data.shape[-1] != self.in_features:
            raise ValueError(
                f"输入特征维度不匹配，期望 {self.in_features}，实际 {x.data.shape[-1]}"
            )
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output
