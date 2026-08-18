from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from KSNet.core.KSangNet import KSNet
from KSNet.core.KSLossBase import KSLossBase
from KSNet.core.KSOptimizerBase import KSOptimizerBase


class KSSequential(KSNet):
    """按添加顺序串联多个层的模型容器。

    与 ``torch.nn.Sequential`` 类似，调用 ``model(x)`` 时会自动完成所有层的
    前向传播。``train_step`` 进一步封装损失计算、反向传播和参数更新，训练
    脚本不再需要逐层手动调用 ``forward`` / ``backward``。
    """

    def __init__(self, *layers: KSNet):
        super().__init__()
        self.trainable = False
        self.layers: list[KSNet] = []
        for layer in layers:
            self.add(layer)

    def add(self, layer: KSNet) -> None:
        """在模型末尾添加一层。"""
        if not isinstance(layer, KSNet):
            raise TypeError(f"layer 必须是 KSNet 实例，实际为 {type(layer).__name__}")
        self.layers.append(layer)
        layer.train() if self.training else layer.eval()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """自动依次完成所有层的前向传播。"""
        self.input = x
        output = x
        for layer in self.layers:
            output = layer(output)
        self.output = output
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """自动按相反顺序完成所有层的反向传播。"""
        if self.output is None:
            raise RuntimeError("请先调用 model(x) 再进行反向传播")

        dx = dout
        for layer in reversed(self.layers):
            dx = layer.backward(dx)
        return dx

    def train_step(
        self,
        x: np.ndarray,
        label: np.ndarray,
        loss_fn: KSLossBase,
        optimizer: KSOptimizerBase,
    ) -> float:
        """完成一次完整训练：前向、计算损失、反向和更新参数。"""
        self.train()
        optimizer.zero_grad()
        prediction = self(x)
        loss = loss_fn.forward(prediction, label)
        self.backward(loss_fn.backward())
        optimizer.step()
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """在推理模式下执行一次前向计算。"""
        self.eval()
        return self(x)

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray | None]]:
        params: list[tuple[np.ndarray, np.ndarray | None]] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self) -> KSSequential:
        self.training = True
        for layer in self.layers:
            layer.train()
        return self

    def eval(self) -> KSSequential:
        self.training = False
        for layer in self.layers:
            layer.eval()
        return self

    def __iter__(self) -> Iterator[KSNet]:
        """让优化器可直接接收 ``layers=model``。"""
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> KSNet:
        return self.layers[index]
