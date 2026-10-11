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
