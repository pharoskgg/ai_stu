from __future__ import annotations

import numpy as np

from KSNet.core.KSangNet import KSNet


class KSBatchNormal(KSNet):
    """对每个特征执行 Batch Normalization。

    输入形状为 ``(batch_size, *input_shape)``，归一化仅沿 batch 维进行。
    ``gamma``/``beta`` 映射到基类的 ``weight``/``bias``，现有优化器因而会
    自动更新这两个参数。
    """

    def __init__(
        self,
        input_shape: int | tuple[int, ...],
        epsilon: float = 1e-8,
        momentum: float = 0.9,
    ) -> None:
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if not input_shape or any(size <= 0 for size in input_shape):
            raise ValueError(f"input_shape 必须是正整数或正整数元组，实际为 {input_shape}")
        if epsilon <= 0:
            raise ValueError(f"epsilon 必须大于 0，实际为 {epsilon}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum 必须在 [0, 1) 内，实际为 {momentum}")

        self.input_shape = input_shape
        self.epsilon = epsilon
        self.momentum = momentum

        # weight/bias 是唯一存储；gamma/beta 是保留 BatchNorm 语义的别名。
        self.weight = np.ones(self.input_shape, dtype=float)
        self.bias = np.zeros(self.input_shape, dtype=float)
        self.w_grad = np.zeros_like(self.weight)
        self.b_grad = np.zeros_like(self.bias)

        self.running_mean = np.zeros(self.input_shape, dtype=float)
        self.running_var = np.ones(self.input_shape, dtype=float)
        self._x_hat: np.ndarray | None = None
        self._inv_std: np.ndarray | None = None

    @property
    def gamma(self) -> np.ndarray:
        return self.weight

    @gamma.setter
    def gamma(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float)
        if value.shape != self.input_shape:
            raise ValueError(f"gamma 形状应为 {self.input_shape}，实际为 {value.shape}")
        self.weight = value
        self.w_grad = np.zeros_like(value)

    @property
    def beta(self) -> np.ndarray:
        return self.bias

    @beta.setter
    def beta(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float)
        if value.shape != self.input_shape:
            raise ValueError(f"beta 形状应为 {self.input_shape}，实际为 {value.shape}")
        self.bias = value
        self.b_grad = np.zeros_like(value)

    @property
    def gamma_grad(self) -> np.ndarray:
        return self.w_grad

    @property
    def beta_grad(self) -> np.ndarray:
        return self.b_grad

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != len(self.input_shape) + 1:
            raise ValueError(
                f"BatchNorm 输入应为 (batch_size, {self.input_shape})，实际为 {x.shape}"
            )
        if x.shape[1:] != self.input_shape:
            raise ValueError(
                f"输入特征形状不匹配，期望 {self.input_shape}，实际为 {x.shape[1:]}"
            )
        if x.shape[0] == 0:
            raise ValueError("BatchNorm 不支持空 batch")

        self.input = x
        if self.training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0) # 计算方差
            inv_std = 1.0 / np.sqrt(var + self.epsilon)
            x_hat = (x - mean) * inv_std

            self.running_mean *= self.momentum
            self.running_mean += (1.0 - self.momentum) * mean
            self.running_var *= self.momentum
            self.running_var += (1.0 - self.momentum) * var
            self._x_hat = x_hat
            self._inv_std = inv_std
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        self.output = self.gamma * x_hat + self.beta
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.input is None or self.output is None or self._x_hat is None or self._inv_std is None:
            raise RuntimeError("请先在训练模式下执行 forward，再调用 backward")
        if dout.shape != self.output.shape:
            raise ValueError(f"上游梯度形状不匹配，期望 {self.output.shape}，实际为 {dout.shape}")

        if self.trainable:
            self.w_grad += np.sum(dout * self._x_hat, axis=0)
            self.b_grad += np.sum(dout, axis=0)

        batch_size = dout.shape[0]
        dx_hat = dout * self.gamma
        return self._inv_std / batch_size * (
            batch_size * dx_hat
            - np.sum(dx_hat, axis=0)
            - self._x_hat * np.sum(dx_hat * self._x_hat, axis=0)
        )
