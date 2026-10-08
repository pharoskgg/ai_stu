"""自动求导的公开入口。"""

from .engine import run_backward
from .function import Context, Function

__all__ = ["Context", "Function", "backward"]


def backward(tensor, gradient=None):
    """对单个输出 Tensor 执行反向传播。"""
    return run_backward(tensor, gradient)
