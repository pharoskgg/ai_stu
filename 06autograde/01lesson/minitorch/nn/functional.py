"""无参数的网络函数接口，连接 Tensor 与底层算子。"""

from ..ops.activation import ReLU, Tanh
from ..ops.loss import CrossEntropy
from ..tensor import Tensor
from ._reduction import resolve_reduction


def relu(input: Tensor) -> Tensor:
    """逐元素取输入和零的最大值，保留输入形状。"""
    return ReLU.apply(input)


def tanh(input: Tensor) -> Tensor:
    """逐元素计算双曲正切，通过 Tanh 算子保留计算图。"""
    return Tanh.apply(input)


def cross_entropy(input: Tensor, target: Tensor, weight=None, size_average=None,
                  ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0) -> Tensor:
    """从 logits 计算交叉熵；类别索引在当前 float64 Tensor 中须为整数值。"""
    for name, value in (("input", input), ("target", target)):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} 必须是 Tensor")
    if weight is not None and not isinstance(weight, Tensor):
        raise TypeError("weight 必须是 Tensor 或 None")
    reduction = resolve_reduction(reduction, size_average, reduce)
    inputs = (input, target) if weight is None else (input, target, weight)
    return CrossEntropy.apply(*inputs, ignore_index=ignore_index, reduction=reduction,
                              label_smoothing=label_smoothing)
