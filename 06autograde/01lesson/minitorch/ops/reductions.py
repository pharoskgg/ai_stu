"""归约运算的前向计算与局部反向公式。"""

from __future__ import annotations
import numpy as np

from ..autograd.function import Context, Function


class Sum(Function):
    @staticmethod
    def _save_metadata(ctx:Context, a, axis, keepdims):
        ctx.input_shape = a.shape
        ctx.keepdims = keepdims
        if axis is None:
            ctx.axes = tuple(range(a.ndim))
        elif a.ndim == 0:
            # NumPy 允许标量使用 axis=0 或 axis=-1。
            ctx.axes = ()
        else:
            axes = (axis,) if isinstance(axis, (int, np.integer)) else axis
            ctx.axes = tuple(int(ax) % a.ndim for ax in axes)

    @staticmethod
    def forward(ctx:Context, a, *, axis=None, keepdims=False):
        result = np.sum(a, axis=axis, keepdims=keepdims)
        Sum._save_metadata(ctx, a, axis, keepdims)
        return result

    @staticmethod
    def backward(ctx:Context, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,)
        if not ctx.keepdims and ctx.axes:
            grad_output = np.expand_dims(grad_output, axis=ctx.axes)
        return (np.broadcast_to(grad_output, ctx.input_shape),)


class Mean(Sum):
    @staticmethod
    def forward(ctx:Context, a, *, axis=None, keepdims=False):
        result = np.mean(a, axis=axis, keepdims=keepdims)
        Sum._save_metadata(ctx, a, axis, keepdims)
        ctx.count = int(np.prod([a.shape[ax] for ax in ctx.axes]))
        return result

    @staticmethod
    def backward(ctx:Context, grad_output):
        return Sum.backward(ctx, grad_output / ctx.count)

