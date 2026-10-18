"""无参数的网络函数接口，连接 Tensor 与底层算子。"""

from ..ops.activation import ReLU, Tanh
from ..tensor import Tensor


def relu(input: Tensor) -> Tensor:
    """逐元素取输入和零的最大值，保留输入形状。"""
    return ReLU.apply(input)


def tanh(input: Tensor) -> Tensor:
    """逐元素计算双曲正切，通过 Tanh 算子保留计算图。"""
    return Tanh.apply(input)
