"""优化器共享的参数管理和梯度清空。"""

from ..tensor import Tensor


class Optimizer:
    def __init__(self, params):
        self.params = []
        seen = set()
        for parameter in params:
            if not isinstance(parameter, Tensor):
                raise TypeError("optimizer parameters must be Tensors")
            if parameter.grad_fn is not None:
                raise ValueError("optimizer parameters must be leaf Tensors")
            if id(parameter) not in seen:
                self.params.append(parameter)
                seen.add(id(parameter))
        if not self.params:
            raise ValueError("optimizer requires at least one parameter")

    def zero_grad(self, set_to_none=True):
        for parameter in self.params:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.fill(0)

    def step(self):
        raise NotImplementedError
