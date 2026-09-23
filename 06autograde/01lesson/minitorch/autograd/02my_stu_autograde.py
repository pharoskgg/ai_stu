from __future__ import annotations
import math

class Context:
    def __init__(self):
        self.saved_values = ()
        self.needs_input_grad = ()
    
    def save_for_backward(self, *values):
        self.saved_values = values


class Function:
    @classmethod
    def apply(cls:Function, *args):
        tensors = tuple(Tensor._ensure_tensor(x) for x in args)

        ctx = Context()
        ctx.needs_input_grad = tuple(tensor.requires_grad for tensor in tensors)

        # 前向传播
        data = cls.forward(ctx, *(tensor.data for tensor in tensors))

        # 被算子创建的Tensor是否需要求梯度
        requires_grad = any(tensor.requires_grad for tensor in tensors)

        grad_fn = None
        if requires_grad:
            grad_fn = FunctionNode(cls, ctx, tensors) # 新创建的Tensor要指明怎么创建的

        return Tensor(data, requires_grad, grad_fn)


class Mul(Function):
    @staticmethod
    def forward(ctx:Context, a:float, b:float):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx:Context, grad_output:float):
        a, b = ctx.saved_values

        grad_a = grad_output * b
        grad_b = grad_output * a

        return grad_a, grad_b

class FunctionNode:
    def __init__(self, function:Function, ctx:Context, parents:tuple[Tensor, ...]):
        self.function = function
        self.ctx = ctx
        self.parents = parents

class Tensor:
    def __init__(self, data:float, requires_grad:bool = False, grad_fn:FunctionNode = None):
        self.data = data
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn

    @staticmethod
    def _ensure_tensor(value):
        if isinstance(value, Tensor):
            return value

        return Tensor(value)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return Mul.apply(other, self)

    def _build_topological_order(self):
        topo = []

        return topo

    def backward(self, grad_output):
        if not self.requires_grad:
            raise RuntimeError(
                "cannot call backward() on a tensor that does not require gradients"
            )

        topo = self._build_topological_order()

        for tensor in reversed(topo):
            pass
        

