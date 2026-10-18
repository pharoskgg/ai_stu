"""网络参数与模块接口。"""

from .parameter import Parameter
from . import functional
from .modules import Module, Linear, ReLU, Tanh, Sequential, RNN

__all__ = ["Parameter", "Module", "Linear", "ReLU", "Tanh", "Sequential", "RNN", "functional"]
