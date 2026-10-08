"""算术运算的前向计算与局部反向公式。"""

from __future__ import annotations
import numpy as np

from ..autograd.function import Context, Function
from ..autograd.utils import sum_to_shape


class Mul(Function):
    @staticmethod
    def forward(ctx:Context, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx:Context, grad_output):
        a, b = ctx.saved_values

        grad_a = (sum_to_shape(grad_output * b, a.shape)
                 if ctx.needs_input_grad[0] else None)

        grad_b = (sum_to_shape(grad_output * a, b.shape)
                 if ctx.needs_input_grad[1] else None)

        return grad_a, grad_b


class Neg(Function):
    @staticmethod
    def forward(ctx:Context, a):
        return -a

    @staticmethod
    def backward(ctx:Context, grad_output):
        return (-grad_output if ctx.needs_input_grad[0] else None,)


class Add(Function):
    @staticmethod
    def forward(ctx:Context, a, b):
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(ctx:Context, grad_output):
        a, b = ctx.saved_values
        grad_a = (sum_to_shape(grad_output, a.shape)
                 if ctx.needs_input_grad[0] else None)

        grad_b = (sum_to_shape(grad_output, b.shape)
                 if ctx.needs_input_grad[1] else None)

        return grad_a, grad_b


class Pow(Function):
    @staticmethod
    def forward(ctx:Context, a, b):
        ctx.save_for_backward(a, b)
        return a ** b

    @staticmethod
    def backward(ctx:Context, grad_out):
        a, b = ctx.saved_values

        grad_a = None
        grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = sum_to_shape(grad_out * b * (a ** (b - 1)), a.shape)

        if ctx.needs_input_grad[1]:
            if np.any(a <= 0):
                raise ValueError("求指数梯度时，底数必须大于 0")

            grad_b = sum_to_shape((grad_out * (a ** b) * np.log(a)), b.shape)

        return grad_a, grad_b
