from __future__ import annotations

import numpy as np

from KSNet.core.KSangNet import KSNet


class KSResidual(KSNet):
    def __init__(
        self,
        main: KSNet,
        shortcut: KSNet | None = None,
        activation: KSNet | None = None,
    ):
        super().__init__()

        if not isinstance(main, KSNet):
            raise TypeError("main 必须是 KSNet 实例")
        if shortcut is not None and not isinstance(shortcut, KSNet):
            raise TypeError("shortcut 必须是 KSNet 实例或 None")
        if activation is not None and not isinstance(activation, KSNet):
            raise TypeError("activation 必须是 KSNet 实例或 None")

        self.main = main
        self.shortcut = shortcut
        self.activation = activation

        # 容器本身没有参数
        self.trainable = False
        self._sum_output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x

        main_output = self.main(x)
        shortcut_output = x if self.shortcut is None else self.shortcut(x)

        if main_output.shape != shortcut_output.shape:
            raise ValueError(
                "残差分支形状不一致："
                f"main={main_output.shape}, shortcut={shortcut_output.shape}。"
                "请使用 Linear 或 1x1 Conv 做投影。"
            )

        self._sum_output = main_output + shortcut_output
        self.output = (
            self._sum_output
            if self.activation is None
            else self.activation(self._sum_output)
        )
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.output is None:
            raise RuntimeError("请先执行 forward 再调用 backward")

        # 先经过相加后的激活层
        dsum = (
            dout
            if self.activation is None
            else self.activation.backward(dout)
        )

        # 加法节点将同一个梯度传给两个分支
        dx_main = self.main.backward(dsum)
        dx_shortcut = (
            dsum
            if self.shortcut is None
            else self.shortcut.backward(dsum)
        )

        # 输入 x 同时流入两个分支，因此输入梯度相加
        return dx_main + dx_shortcut

    def zero_grad(self) -> None:
        self.main.zero_grad()
        if self.shortcut is not None:
            self.shortcut.zero_grad()
        if self.activation is not None:
            self.activation.zero_grad()

    def parameters(self):
        params = list(self.main.parameters())
        if self.shortcut is not None:
            params.extend(self.shortcut.parameters())
        if self.activation is not None:
            params.extend(self.activation.parameters())
        return params

    def train(self):
        self.training = True
        self.main.train()
        if self.shortcut is not None:
            self.shortcut.train()
        if self.activation is not None:
            self.activation.train()
        return self

    def eval(self):
        self.training = False
        self.main.eval()
        if self.shortcut is not None:
            self.shortcut.eval()
        if self.activation is not None:
            self.activation.eval()
        return self