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
