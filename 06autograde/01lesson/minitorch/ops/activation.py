"""激活运算的前向计算与局部反向公式。"""

import numpy as np

from ..autograd.function import Context, Function


class ReLU(Function):
    """负值置零，零点的局部梯度取零。"""

    @staticmethod
    def forward(ctx: Context, input):
        ctx.save_for_backward(input > 0)
        return np.maximum(input, 0)

    @staticmethod
    def backward(ctx: Context, grad_output):
        positive_mask, = ctx.saved_values
        return (grad_output * positive_mask if ctx.needs_input_grad[0] else None,)


class Tanh(Function):
    """保存 tanh 前向结果，局部导数为 1 - output**2。"""

    @staticmethod
    def forward(ctx: Context, input):
        output = np.tanh(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output):
        output, = ctx.saved_values
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * (1 - output ** 2)
        return (grad_input,)
