"""通过 functional 接口实现无参数激活层。"""

from ...tensor import Tensor
from .. import functional
from .module import Module


class ReLU(Module):
    """线性整流层；零点梯度取零，不修改输入。"""

    def forward(self, input: Tensor) -> Tensor:
        return functional.relu(input)
