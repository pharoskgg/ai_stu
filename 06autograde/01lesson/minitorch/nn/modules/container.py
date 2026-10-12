"""复用模块注册机制，按注册顺序串联网络层。"""

from collections import OrderedDict
from numbers import Integral

from .module import Module


class Sequential(Module):
    """接受位置 Module 或单个 OrderedDict；空容器直接返回输入。"""

    def __init__(self, *modules):
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            named_modules = modules[0].items()
        else:
            named_modules = ((str(index), module) for index, module in enumerate(modules))
        for name, module in named_modules:
            if not isinstance(module, Module):
                raise TypeError("Sequential 的元素必须是 Module")
            self.add_module(name, module)

    def forward(self, input):
        # 同一个模块可以注册在多个位置，每个位置都必须执行。
        for module in self._modules.values():
            input = module(input)
        return input

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("Sequential 索引必须是整数")
        return tuple(self._modules.values())[index]
