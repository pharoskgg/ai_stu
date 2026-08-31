from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional, Tuple

import numpy as np


# 兼容 Python 3.9；TypeAlias 和 ``X | None`` 需要更高版本支持。
KSParameter = Tuple[np.ndarray, Optional[np.ndarray]]

class KSOptimizerBase(ABC):
    """优化器统一抽象基类。

    与 PyTorch 的 ``Optimizer(model.parameters(), ...)`` 接口一致，优化器只
    管理参数和梯度，不关心参数位于普通层还是 Sequential/Residual 容器中。
    """

    def __init__(
        self,
        params: Iterable[KSParameter],
        lr: float,
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        for index, item in enumerate(self.params):
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "优化器需要接收 model.parameters()，其中每一项应为 "
                    f"(param, grad)，第 {index} 项实际为 {type(item).__name__}"
                )

            param, grad = item
            if not isinstance(param, np.ndarray):
                raise TypeError(f"第 {index} 个参数必须是 np.ndarray")
            if grad is not None and not isinstance(grad, np.ndarray):
                raise TypeError(f"第 {index} 个梯度必须是 np.ndarray 或 None")
            if grad is not None and param.shape != grad.shape:
                raise ValueError(
                    f"第 {index} 个参数与梯度形状不一致：{param.shape} != {grad.shape}"
                )

        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        """直接清空优化器管理的全部梯度。"""
        for _, grad in self.params:
            if grad is not None:
                grad.fill(0.0)

    @abstractmethod
    def step(self) -> None:
        """参数更新核心逻辑，各优化器自行实现"""
        pass

    def _get_trainable_params(self):
        """遍历优化器管理的参数，只返回梯度有效的参数。"""
        for param, grad in self.params:
            if grad is not None:
                yield param, grad
