from __future__ import annotations
import numpy as np

def sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """将广播结果的梯度求和回原输入形状。"""
    grad = np.asarray(grad)

    # 消除前向广播新增的前导维度
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # 原来长度为 1 的维度，在反向时求和
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Context:
    def __init__(self):
        self.saved_values = ()

        self.needs_input_grad = ()

    def save_for_backward(self, *values):

        self.saved_values = tuple(
            value.copy() if isinstance(value, np.ndarray) else value for value in values
        )

class Function:
    @classmethod
    def apply(cls:type[Function], *args):
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
    def forward(ctx:Context, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx:Context, grad_output):
        a, b = ctx.saved_values

        grad_a = (sum_to_shape(grad_output * b, a.shape)
                 if ctx.needs_input_grad[0] else None)

        grad_b = (sum_to_shape(grad_output * a, b.shape)
                 if ctx.needs_input_grad[1] else None)

        return grad_a, grad_b

class Neg(Function):
    @staticmethod
    def forward(ctx:Context, a):
        return -a

    @staticmethod
    def backward(ctx:Context, grad_output):
        return (-grad_output if ctx.needs_input_grad[0] else None,)

class Add(Function):
    @staticmethod
    def forward(ctx:Context, a, b):
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(ctx:Context, grad_output):
        a, b = ctx.saved_values
        grad_a = (sum_to_shape(grad_output, a.shape)
                 if ctx.needs_input_grad[0] else None)

        grad_b = (sum_to_shape(grad_output, b.shape)
                 if ctx.needs_input_grad[1] else None)

        return grad_a, grad_b

class Pow(Function):
    @staticmethod
    def forward(ctx:Context, a, b):
        ctx.save_for_backward(a, b)
        return a ** b

    @staticmethod
    def backward(ctx:Context, grad_out):
        a, b = ctx.saved_values

        grad_a = None
        grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = sum_to_shape(grad_out * b * (a ** (b - 1)), a.shape)

        if ctx.needs_input_grad[1]:
            if np.any(a <= 0):
                raise ValueError("求指数梯度时，底数必须大于 0")

            grad_b = sum_to_shape((grad_out * (a ** b) * np.log(a)), b.shape)

        return grad_a, grad_b

class MatMul(Function):
    @staticmethod
    def forward(ctx:Context, a, b):
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("MatMul 当前要求两个输入至少为二维")
        ctx.save_for_backward(a, b)
        return np.matmul(a, b)

    @staticmethod
    def backward(ctx:Context, grad_output):
        a, b = ctx.saved_values
        grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            grad_a = np.matmul(grad_output, np.swapaxes(b, -1, -2))
            grad_a = sum_to_shape(grad_a, a.shape)

        if ctx.needs_input_grad[1]:
            grad_b = np.matmul(np.swapaxes(a, -1, -2), grad_output)
            grad_b = sum_to_shape(grad_b, b.shape)

        return grad_a, grad_b

class FunctionNode:
    def __init__(self, function:type[Function], ctx:Context, parents:tuple[Tensor, ...]):
        self.function = function
        self.ctx = ctx
        self.parents = parents

    def backward(self, grad_output:np.ndarray):
        return self.function.backward(self.ctx, grad_output)


class Tensor:
    def __init__(self, data, requires_grad:bool = False, grad_fn:FunctionNode | None = None):
        self.data: np.ndarray = np.array(data, dtype=np.float64, copy=True)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad: np.ndarray | None = None

    @staticmethod
    def _ensure_tensor(value):
        if isinstance(value, Tensor):
            return value

        return Tensor(value)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return Mul.apply(other, self)

    def __matmul__(self, other):
        return MatMul.apply(self, other)

    def __rmatmul__(self, other):
        return MatMul.apply(other, self)

    def __add__(self, other):
        return Add.apply(self, other)

    def __radd__(self, other):
        return Add.apply(other, self)

    def __neg__(self):
        return Neg.apply(self)

    def __sub__(self, other):
        return self + (-Tensor._ensure_tensor(other))

    def __rsub__(self, other):
        return (Tensor._ensure_tensor(other) + (-self))

    def __pow__(self, other):
        return Pow.apply(self, other)

    def __rpow__(self, other):
        return Pow.apply(other, self)

    # 除法
    def __truediv__(self, other):
        other = Tensor._ensure_tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = Tensor._ensure_tensor(other)
        return other * (self ** -1)

    def _build_topological_order(self):
        topo = []
        visited = set()

        def visit(tensor:Tensor):
            if tensor in visited:
                return

            visited.add(tensor)
            # 不是叶子节点才有parents
            if tensor.grad_fn is not None:
                for parent in tensor.grad_fn.parents:
                    visit(parent)

            topo.append(tensor)

        visit(self)

        return topo

    def backward(self, gradient=None):
        if not self.requires_grad:
            raise RuntimeError(
                "cannot call backward() on a tensor that does not require gradients"
            )
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("多元素输出调用 backward() 时需要提供 gradient")
            gradient = np.ones_like(self.data)
        else:
            gradient = np.asarray(gradient, dtype=self.data.dtype)
            if gradient.shape != self.data.shape:
                raise ValueError(f"gradient 形状为 {gradient.shape}，输出形状为 {self.data.shape}")

        topo = self._build_topological_order()

        gradients = {id(self): gradient}

        for tensor in reversed(topo):
            tensor_id = id(tensor)

            if tensor_id not in gradients:
                continue

            grad_output = gradients[tensor_id]

            # leaf
            if tensor.grad_fn is None:
                if tensor.requires_grad:
                    if tensor.grad is None:
                        tensor.grad = grad_output
                    else:
                        tensor.grad += grad_output
                continue

            grad_inputs = (tensor.grad_fn.backward(grad_output))

            if not isinstance(grad_inputs, tuple):
                grad_inputs = (grad_inputs,)
            # 梯度数量返回检查
            if len(grad_inputs) != len(tensor.grad_fn.parents):
                raise RuntimeError(
                    f"{tensor.grad_fn.function.__name__}.backward returned "
                    f"{len(grad_inputs)} gradients, but expected "
                    f"{len(tensor.grad_fn.parents)}"
                )

            for parent, grad_input in zip(tensor.grad_fn.parents, grad_inputs):
                # 可能有些tensor设置为不进行梯度计算，返回None
                if grad_input is None:
                    continue

                if not parent.requires_grad:
                    continue

                parent_id = id(parent)

                # 临时累计梯度到父节点中
                if parent_id not in gradients:
                    gradients[parent_id] = (grad_input)
                else:
                    gradients[parent_id] += (grad_input)

