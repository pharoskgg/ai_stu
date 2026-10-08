"""基于 NumPy 的教学用自动求导包。"""

from .tensor import Tensor
from . import nn

__all__ = ["Tensor", "nn"]
