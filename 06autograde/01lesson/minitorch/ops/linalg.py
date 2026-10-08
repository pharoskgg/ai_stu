"""矩阵运算的前向计算与局部反向公式。"""

from __future__ import annotations
import numpy as np

from ..autograd.function import Context, Function
from ..autograd.utils import sum_to_shape


class MatMul(Function):
    @staticmethod
    def forward(ctx:Context, a, b):
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("MatMul 当前要求两个输入至少为二维")
        ctx.save_for_backward(a, b)
        return np.matmul(a, b)

    @staticmethod
    def backward(ctx:Context, grad_output):
        a, b = ctx.saved_values
        grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            grad_a = np.matmul(grad_output, np.swapaxes(b, -1, -2))
            grad_a = sum_to_shape(grad_a, a.shape)

        if ctx.needs_input_grad[1]:
            grad_b = np.matmul(np.swapaxes(a, -1, -2), grad_output)
            grad_b = sum_to_shape(grad_b, b.shape)

        return grad_a, grad_b
