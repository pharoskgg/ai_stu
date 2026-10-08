"""Tensor 的数据、梯度状态和用户运算接口。"""

from __future__ import annotations
import numpy as np

from .autograd.graph import FunctionNode
from .ops.arithmetic import Add, Mul, Neg, Pow
from .ops.linalg import MatMul


class Tensor:
    def __init__(self, data, requires_grad:bool = False, grad_fn:FunctionNode | None = None):
        self.data: np.ndarray = np.array(data, dtype=np.float64, copy=True)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad: np.ndarray | None = None

    @staticmethod
    def _ensure_tensor(value):
        if isinstance(value, Tensor):
            return value

        return Tensor(value)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return Mul.apply(other, self)

    def __matmul__(self, other):
        return MatMul.apply(self, other)

    def __rmatmul__(self, other):
        return MatMul.apply(other, self)

    def __add__(self, other):
        return Add.apply(self, other)

    def __radd__(self, other):
        return Add.apply(other, self)

    def __neg__(self):
        return Neg.apply(self)

    def __sub__(self, other):
        return self + (-Tensor._ensure_tensor(other))

    def __rsub__(self, other):
        return (Tensor._ensure_tensor(other) + (-self))

    def __pow__(self, other):
        return Pow.apply(self, other)

    def __rpow__(self, other):
        return Pow.apply(other, self)

    # 除法
    def __truediv__(self, other):
        other = Tensor._ensure_tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = Tensor._ensure_tensor(other)
        return other * (self ** -1)

    def zero_grad(self):
        self.grad = None


    def backward(self, gradient=None):
        # Tensor 提供入口，整张图的执行交给 autograd。
        from .autograd import backward

        return backward(self, gradient)
