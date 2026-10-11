"""网络参数与模块接口。"""

from .parameter import Parameter
from . import functional
from .modules import Module, Linear, ReLU

__all__ = ["Parameter", "Module", "Linear", "ReLU", "functional"]
