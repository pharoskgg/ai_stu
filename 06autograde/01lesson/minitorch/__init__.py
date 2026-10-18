"""基于 NumPy 的教学用自动求导包。"""

from .tensor import Tensor, stack
from . import nn, optim

__all__ = ["Tensor", "stack", "nn", "optim"]
