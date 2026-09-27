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
    def forward(ctx:Context, a:float, b:float):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx:Context, grad_output:float):
        a, b = ctx.saved_values

        grad_a = grad_output * b if ctx.needs_input_grad[0] else None
        grad_b = grad_output * a if ctx.needs_input_grad[1] else None

        return grad_a, grad_b

class FunctionNode:
    def __init__(self, function:type[Function], ctx:Context, parents:tuple[Tensor, ...]):
        self.function = function
        self.ctx = ctx
        self.parents = parents

    def backward(self, grad_output):
        return self.function.backward(self.ctx, grad_output)
    

class Tensor:
    def __init__(self, data:float, requires_grad:bool = False, grad_fn:FunctionNode = None):
        self.data = data
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None

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
            gradient = 1.0

        gradient = float(gradient)
        
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
            
        

