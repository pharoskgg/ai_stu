"""每次运算对应的反向节点及其父 Tensor 连接。"""

from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tensor import Tensor
    from .function import Context, Function


class FunctionNode:
    def __init__(self, function:type[Function], ctx:Context, parents:tuple[Tensor, ...]):
        self.function = function
        self.ctx = ctx
        self.parents = parents

    def backward(self, grad_output:np.ndarray):
        return self.function.backward(self.ctx, grad_output)
