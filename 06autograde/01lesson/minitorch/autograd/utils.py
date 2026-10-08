"""反向传播使用的形状工具。"""

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
