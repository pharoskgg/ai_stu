"""遍历计算图、执行反向传播并累加叶子梯度。

函数参数 self 沿用原代码命名，表示反向传播的输出 Tensor。
"""

from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tensor import Tensor


def _build_topo_with_stack(self):
    topo = []
    visited = set()
    stack = [(self, False)]

    while stack:
        tensor, expanded = stack.pop()

        # 父节点处理完毕，此时加入拓扑
        if expanded:
            topo.append(tensor)
            continue

        if tensor in visited:
            continue

        visited.add(tensor)

        # 栈先进后出: 先处理父节点，再处理当前节点
        stack.append((tensor, True))

        if tensor.grad_fn is not None:
            for parent in reversed(tensor.grad_fn.parents):
                if parent not in visited:
                    stack.append((parent, False))

    return topo


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


def run_backward(self, gradient=None):
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

    topo = _build_topo_with_stack(self)

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
                    tensor.grad = grad_output.copy()
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

            # 检查 backward 是否返回了正确形状
            if grad_input.shape != parent.data.shape:
                raise RuntimeError(
                    f"{tensor.grad_fn.function.__name__}.backward 返回梯度形状 "
                    f"{grad_input.shape}，但输入形状是 {parent.data.shape}"
                )

            parent_id = id(parent)

            # 临时累计梯度到父节点中
            if parent_id not in gradients:
                gradients[parent_id] = (grad_input.copy())
            else:
                gradients[parent_id] += (grad_input)
