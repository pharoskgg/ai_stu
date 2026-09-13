from KSNet.core.KSangNet import KSNet
import numpy as np
from typing import Optional

class KSRnn(KSNet):
    def __init__(self, input_size: int, hidden_size: int, return_sequences: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.trainable = True

        self.w_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.w_x = np.random.randn(hidden_size, input_size) * 0.01
        self.b = np.zeros((hidden_size, 1))

        self.dw_x = np.zeros_like(self.w_x)
        self.dw_h = np.zeros_like(self.w_h)
        self.db = np.zeros_like(self.b)

        # 内部统一使用 (seq_len + 1, batch_size, hidden_size)。
        self.hidden_states = None
        # loss 对 h0 的梯度，Seq2Seq 中用它将 decoder 梯度传给 encoder。
        self.dh_initial = None
        self.mask = None
        self._batched_input = False

    def forward(
        self,
        x: np.ndarray,
        h_prev=None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """支持单条 ``(T, D)`` 和 batch ``(B, T, D)`` 输入。

        ``mask`` 形状为 ``(T,)`` 或 ``(B, T)``，True 表示有效时间步。
        mask 为 False 时直接保留上一个隐藏状态，避免 padding 改变结果。
        """
        x = np.asarray(x)
        if x.ndim not in (2, 3):
            raise ValueError(f"RNN输入必须是 (T, D) 或 (B, T, D)，实际为 {x.shape}")
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"输入特征维度应为 {self.input_size}，实际为 {x.shape[-1]}"
            )

        self._batched_input = x.ndim == 3
        batch_inputs = x if self._batched_input else x[None, ...]
        batch_size, sequence_length, _ = batch_inputs.shape
        if sequence_length == 0:
            raise ValueError("序列长度不能为 0")

        if h_prev is None:
            initial_hidden = np.zeros((batch_size, self.hidden_size))
        else:
            initial_hidden = np.asarray(h_prev)
            if not self._batched_input and initial_hidden.size == self.hidden_size:
                initial_hidden = initial_hidden.reshape(1, self.hidden_size)
            if initial_hidden.shape != (batch_size, self.hidden_size):
                raise ValueError(
                    "h_prev 形状应为 "
                    f"{(batch_size, self.hidden_size)}，实际为 {initial_hidden.shape}"
                )

        if mask is None:
            batch_mask = np.ones((batch_size, sequence_length), dtype=bool)
        else:
            batch_mask = np.asarray(mask, dtype=bool)
            if not self._batched_input and batch_mask.shape == (sequence_length,):
                batch_mask = batch_mask[None, :]
            if batch_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    f"mask 形状应为 {(batch_size, sequence_length)}，"
                    f"实际为 {batch_mask.shape}"
                )

        self.inputs = batch_inputs
        self.mask = batch_mask
        hidden_states = [initial_hidden.copy()]
        hidden = initial_hidden
        bias = self.b.reshape(1, self.hidden_size)
        for t in range(sequence_length):
            candidate = np.tanh(
                hidden @ self.w_h.T + batch_inputs[:, t, :] @ self.w_x.T + bias
            )
            active = batch_mask[:, t, None]
            hidden = np.where(active, candidate, hidden)
            hidden_states.append(hidden)

        self.hidden_states = np.stack(hidden_states, axis=0)
        sequence_output = np.swapaxes(self.hidden_states[1:], 0, 1)  # (B, T, H)
        output = sequence_output if self.return_sequences else self.hidden_states[-1]
        self.output = output if self._batched_input else output[0]
        return self.output

    def backward(self, dout:np.ndarray):
        dout = np.asarray(dout)
        if self.hidden_states is None:
            raise RuntimeError("请先执行 forward，再执行 backward")

        batch_size, sequence_length, _ = self.inputs.shape

        if self.return_sequences:
            expected_shape = (
                (batch_size, sequence_length, self.hidden_size)
                if self._batched_input
                else (sequence_length, self.hidden_size)
            )
            if dout.shape != expected_shape:
                raise ValueError(
                    f"dout形状应为{expected_shape}，实际为{dout.shape}"
                )
            direct_dh = dout if self._batched_input else dout[None, ...]
        else:
            expected_shape = (
                (batch_size, self.hidden_size)
                if self._batched_input
                else (self.hidden_size,)
            )
            if dout.shape != expected_shape:
                raise ValueError(
                    f"dout形状应为{expected_shape}，实际为{dout.shape}"
                )
            direct_dh = np.zeros(
                (batch_size, sequence_length, self.hidden_size),
                dtype=np.result_type(dout.dtype, self.inputs.dtype),
            )
            direct_dh[:, -1, :] = dout if self._batched_input else dout[None, :]

        gradient_dtype = np.result_type(dout.dtype, self.inputs.dtype)
        dh_next = np.zeros((batch_size, self.hidden_size), dtype=gradient_dtype)
        d_inputs = np.zeros_like(self.inputs, dtype=gradient_dtype)
        for t in range(sequence_length - 1, -1, -1):
            h_t = self.hidden_states[t + 1]
            h_prev = self.hidden_states[t]

            dh = direct_dh[:, t, :] + dh_next
            active = self.mask[:, t, None]
            dz = dh * (1.0 - h_t ** 2) * active

            self.dw_h += dz.T @ h_prev
            self.dw_x += dz.T @ self.inputs[:, t, :]
            self.db += np.sum(dz, axis=0).reshape(self.hidden_size, 1)

            # padding 时 h_t = h_prev，因此梯度沿恒等连接传回。
            dh_next = dz @ self.w_h + dh * (~active)
            d_inputs[:, t, :] = dz @ self.w_x

        self.dh_initial = dh_next if self._batched_input else dh_next[0]
        return d_inputs if self._batched_input else d_inputs[0]

    def parameters(self):
        if not self.trainable:
            return []
        return [(self.w_x, self.dw_x), 
                (self.w_h, self.dw_h),
                (self.b, self.db)]

    def zero_grad(self):
        self.dw_x.fill(0.0)
        self.dw_h.fill(0.0)
        self.db.fill(0.0)
