"""保存计算上下文，并在前向执行时建立反向图。"""

from __future__ import annotations
import numpy as np

from .graph import FunctionNode


class Context:
    def __init__(self):
        self.saved_values = ()

        self.needs_input_grad = ()

    def save_for_backward(self, *values):

        self.saved_values = tuple(
            value.copy() if isinstance(value, np.ndarray) else value for value in values
        )


class Function:
    @classmethod
    def apply(cls:type[Function], *args):
        # 延迟导入，避免 Tensor → 算子 → Function → Tensor 循环导入。
        from ..tensor import Tensor

        tensors = tuple(Tensor._ensure_tensor(x) for x in args)

        ctx = Context()
        ctx.needs_input_grad = tuple(tensor.requires_grad for tensor in tensors)

        # 前向传播
        data = cls.forward(ctx, *(tensor.data for tensor in tensors))

        # 被算子创建的Tensor是否需要求梯度
        requires_grad = any(tensor.requires_grad for tensor in tensors)

        grad_fn = None
        if requires_grad:
            grad_fn = FunctionNode(cls, ctx, tensors) # 新创建的Tensor要指明怎么创建的

        return Tensor(data, requires_grad, grad_fn)
