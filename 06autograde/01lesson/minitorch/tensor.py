"""Tensor 的数据、梯度状态和用户运算接口。"""

from __future__ import annotations
import numpy as np

from .autograd.graph import FunctionNode
from .ops.arithmetic import Add, Mul, Neg, Pow
from .ops.linalg import MatMul
from .ops.reductions import Sum, Mean
from .ops.shape import Reshape, Transpose, Select, Stack, _normalize_dim


class Tensor:
    def __init__(self, data, requires_grad:bool = False, grad_fn:FunctionNode | None = None):
        self.data: np.ndarray = np.array(data, dtype=np.float64, copy=True)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad: np.ndarray | None = None

    @property
    def shape(self):
        """返回形状元组。"""
        return self.data.shape

    @property
    def ndim(self):
        """返回维度数量，标量为零维。"""
        return self.data.ndim

    def size(self, dim=None):
        """返回完整形状或指定维度的长度，支持负维度。"""
        if dim is None:
            return self.shape
        return self.shape[_normalize_dim(dim, self.ndim)]

    def numel(self):
        """返回元素总数。"""
        return self.data.size

    def reshape(self, *shape):
        """改变形状，接受多个整数或一个元组/列表，允许一个 -1。"""
        if not shape:
            raise TypeError("reshape 至少需要一个形状参数；标量使用 reshape(())")
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Reshape.apply(self, shape=shape)

    def transpose(self, dim0, dim1):
        """交换两个维度，反向传播时将梯度交换回原维度。"""
        return Transpose.apply(self, dim0=dim0, dim1=dim1)

    def select(self, dim, index):
        """沿 dim 选取 index 处的切片并移除该维度，支持负编号。"""
        return Select.apply(self, dim=dim, index=index)

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

    def sum(self, dim=None, keepdim=False):
        return Sum.apply(self, axis=dim, keepdims=keepdim)


    def mean(self, dim=None, keepdim=False):
        return Mean.apply(self, axis=dim, keepdims=keepdim)


    def zero_grad(self):
        self.grad = None


    def backward(self, gradient=None):
        # Tensor 提供入口，整张图的执行交给 autograd。
        from .autograd import backward

        return backward(self, gradient)


def stack(tensors, dim=0):
    """沿新增维度堆叠非空 Tensor 列表或元组，保留计算图。"""
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("tensors 必须是 Tensor 列表或元组")
    if any(not isinstance(tensor, Tensor) for tensor in tensors):
        raise TypeError("stack 的每个输入必须是 Tensor")
    return Stack.apply(*tensors, dim=dim)
