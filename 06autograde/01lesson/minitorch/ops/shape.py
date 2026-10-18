"""形状变换的前向计算与局部反向公式。"""

from numbers import Integral

import numpy as np

from ..autograd.function import Context, Function


def _normalize_dim(dim, ndim):
    """校验维度编号，并将负编号转换为非负编号。"""
    if isinstance(dim, bool) or not isinstance(dim, Integral):
        raise TypeError("dim 必须是整数")
    dim = int(dim)
    if not -ndim <= dim < ndim:
        raise IndexError(f"维度 {dim} 超出 {ndim} 维 Tensor 的范围")
    return dim % ndim


class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, a, *, shape):
        if not isinstance(shape, (tuple, list)):
            raise TypeError("shape 必须是整数元组或列表")
        if any(isinstance(size, bool) or not isinstance(size, Integral) for size in shape):
            raise TypeError("shape 中的每个维度必须是整数")
        shape = tuple(int(size) for size in shape)
        if any(size < -1 for size in shape):
            raise RuntimeError("shape 的维度不能小于 -1")
        if shape.count(-1) > 1:
            raise RuntimeError("shape 最多允许一个 -1 推导维度")
        ctx.input_shape = a.shape
        try:
            return np.reshape(a, shape)
        except ValueError as error:
            raise RuntimeError(f"无法将形状 {a.shape} 的 Tensor reshape 为 {shape}") from error

    @staticmethod
    def backward(ctx: Context, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,)
        return (np.reshape(grad_output, ctx.input_shape),)


class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a, *, dim0, dim1):
        # 标量允许 0 和 -1，交换它们仍是标量。
        ctx.dim0 = _normalize_dim(dim0, max(a.ndim, 1))
        ctx.dim1 = _normalize_dim(dim1, max(a.ndim, 1))
        ctx.is_scalar = a.ndim == 0
        if ctx.is_scalar:
            return a
        return np.swapaxes(a, ctx.dim0, ctx.dim1)

    @staticmethod
    def backward(ctx: Context, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,)
        if ctx.is_scalar:
            return (grad_output,)
        return (np.swapaxes(grad_output, ctx.dim0, ctx.dim1),)


class Select(Function):
    """沿一个维度选取一个位置，反向将梯度写回对应切片。"""

    @staticmethod
    def forward(ctx: Context, a, *, dim, index):
        ctx.dim = _normalize_dim(dim, a.ndim)
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("index 必须是整数")
        size = a.shape[ctx.dim]
        if not -size <= index < size:
            raise IndexError(f"index {index} 超出长度 {size} 的范围")
        ctx.index = int(index) % size
        ctx.input_shape = a.shape
        return np.take(a, ctx.index, axis=ctx.dim)

    @staticmethod
    def backward(ctx: Context, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,)
        grad_input = np.zeros(ctx.input_shape, dtype=grad_output.dtype)
        slices = [slice(None)] * len(ctx.input_shape)
        slices[ctx.dim] = ctx.index
        grad_input[tuple(slices)] = grad_output
        return (grad_input,)


class Stack(Function):
    """沿新增维度堆叠输入，反向按相同维度拆分梯度。"""

    @staticmethod
    def forward(ctx: Context, *inputs, dim=0):
        if not inputs:
            raise ValueError("stack 至少需要一个 Tensor")
        ctx.dim = _normalize_dim(dim, inputs[0].ndim + 1)
        if any(a.shape != inputs[0].shape for a in inputs):
            raise ValueError("stack 的所有输入形状必须一致")
        return np.stack(inputs, axis=ctx.dim)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return tuple(
            np.take(grad_output, index, axis=ctx.dim) if needs_grad else None
            for index, needs_grad in enumerate(ctx.needs_input_grad)
        )
