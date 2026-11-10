from __future__ import annotations
import numpy as np

from ..autograd.function import Context, Function

class Embedding(Function):
    @staticmethod
    def forward(ctx: Context, input, weight):
        if weight.ndim != 2:
            raise ValueError("Embedding 当前要求权重为二维张量")

        if not np.issubdtype(input.dtype, np.integer):
            raise TypeError("input 必须是整数类型")

        num_embeddings = weight.shape[0]
        if np.any(input < 0) or np.any(input >= num_embeddings):
            raise ValueError("input 中的索引超出范围")

        ctx.save_for_backward(input, weight)
        return weight[input]

    @staticmethod
    def backward(ctx: Context, grad_output):
        input, weight = ctx.saved_values
        grad_weight = None

        if ctx.needs_input_grad[1]:
            grad_weight = np.zeros_like(weight)
            np.add.at(grad_weight, input, grad_output)

        return None, grad_weight
