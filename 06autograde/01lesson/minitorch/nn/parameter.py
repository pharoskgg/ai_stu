"""可由 Module 注册和管理的叶子 Tensor。"""

import numpy as np

from ..tensor import Tensor


class Parameter(Tensor):
    """默认需要梯度；复制输入数据，不继承输入 Tensor 的计算图。"""

    def __init__(self, data=None, requires_grad: bool = True):
        if data is None:
            data = np.empty(0, dtype=np.float64)
        elif isinstance(data, Tensor):
            data = data.data
        super().__init__(data, requires_grad=requires_grad)

    def __repr__(self):
        return f"Parameter(data={self.data!r}, requires_grad={self.requires_grad})"
