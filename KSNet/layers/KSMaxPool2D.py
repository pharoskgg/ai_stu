from KSNet.core.KSangNet import KSNet
import numpy as np


class KSMaxPool2D(KSNet):
    def __init__(self, kernel_size=2, stride=1, padding=0):
        super().__init__()

        self.kernel_size = self._to_pair(kernel_size, "kernel_size")
        self.stride = self._to_pair(stride, "stride")
        self.padding = self._to_pair(padding, "padding")

        # MaxPool 没有可训练参数
        self.trainable = False

        # 保存前向传播中最大值的位置
        self.max_indices = None
        self.padded_shape = None

        self._validate_parameters()

    @staticmethod
    def _to_pair(value, name):
        """把整数转换成二维元组。"""
        if isinstance(value, (int, np.integer)):
            return int(value), int(value)

        if isinstance(value, (tuple, list)) and len(value) == 2:
            return int(value[0]), int(value[1])

        raise TypeError(
            f"{name} 必须是整数或者长度为 2 的元组，实际为 {value}"
        )

    def _validate_parameters(self):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        if kh <= 0 or kw <= 0:
            raise ValueError("kernel_size 必须大于 0")

        if sh <= 0 or sw <= 0:
            raise ValueError("stride 必须大于 0")

        if ph < 0 or pw < 0:
            raise ValueError("padding 不能小于 0")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播。

        参数:
            x: 输入，形状为 (N, C, H, W)

        返回:
            池化结果，形状为 (N, C, out_h, out_w)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("输入必须是 np.ndarray")

        if x.ndim != 4:
            raise ValueError(
                f"KSMaxPool2D 只支持 NCHW 四维输入，实际形状为 {x.shape}"
            )

        if not (
            np.issubdtype(x.dtype, np.floating)
            or np.issubdtype(x.dtype, np.integer)
        ):
            raise TypeError(f"不支持的输入类型：{x.dtype}")

        self.input = x

        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                "池化窗口大于 padding 后的输入尺寸："
                f"input={x.shape}, kernel_size={self.kernel_size}, "
                f"padding={self.padding}"
            )

        # 浮点数用负无穷，整数用该类型的最小值
        if np.issubdtype(x.dtype, np.floating):
            padding_value = -np.inf
        else:
            padding_value = np.iinfo(x.dtype).min

        x_padded = np.pad(
            x,
            (
                (0, 0),
                (0, 0),
                (ph, ph),
                (pw, pw),
            ),
            mode="constant",
            constant_values=padding_value,
        )

        self.padded_shape = x_padded.shape

        output = np.empty(
            (n, c, out_h, out_w),
            dtype=x.dtype,
        )

        # 每个元素记录其所在窗口中的展平索引
        self.max_indices = np.empty(
            (n, c, out_h, out_w),
            dtype=np.int64,
        )

        for i in range(out_h):
            h_start = i * sh
            h_end = h_start + kh

            for j in range(out_w):
                w_start = j * sw
                w_end = w_start + kw

                # window: (N, C, kh, kw)
                window = x_padded[
                    :,
                    :,
                    h_start:h_end,
                    w_start:w_end,
                ]

                # 转成 (N, C, kh * kw)
                window_flat = window.reshape(n, c, -1)

                # 找到每个窗口最大值的位置
                max_index = np.argmax(window_flat, axis=2)

                # 根据最大值位置取出最大值
                max_value = np.take_along_axis(
                    window_flat,
                    max_index[..., None],
                    axis=2,
                ).squeeze(axis=2)

                output[:, :, i, j] = max_value
                self.max_indices[:, :, i, j] = max_index

        self.output = output
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播。

        参数:
            dout: 上游梯度，形状与 self.output 相同

        返回:
            dx: 输入梯度，形状与 self.input 相同
        """
        if (
            self.input is None
            or self.output is None
            or self.max_indices is None
        ):
            raise RuntimeError("请先执行 forward，再调用 backward")

        if dout.shape != self.output.shape:
            raise ValueError(
                f"上游梯度形状不匹配，期望 {self.output.shape}，"
                f"实际为 {dout.shape}"
            )

        n, c, h, w = self.input.shape
        _, _, out_h, out_w = dout.shape

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        dx_padded = np.zeros(
            self.padded_shape,
            dtype=dout.dtype,
        )

        # 用于同时索引所有 batch 和 channel
        batch_index = np.arange(n)[:, None]
        channel_index = np.arange(c)[None, :]

        for i in range(out_h):
            h_start = i * sh

            for j in range(out_w):
                w_start = j * sw

                max_index = self.max_indices[:, :, i, j]

                # 展平索引转换成窗口内部的行列坐标
                max_row = max_index // kw
                max_col = max_index % kw

                # 使用 np.add.at，因为重叠窗口的梯度需要累加
                np.add.at(
                    dx_padded,
                    (
                        batch_index,
                        channel_index,
                        h_start + max_row,
                        w_start + max_col,
                    ),
                    dout[:, :, i, j],
                )

        # 去掉 padding 部分
        dx = dx_padded[
            :,
            :,
            ph:ph + h,
            pw:pw + w,
        ]

        return dx