from KSNet.core.KSangNet import KSNet
import numpy as np

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

        # 包含初始状态 h0，形状为 (seq_len + 1, hidden_size, 1)
        self.hidden_states = None
        # loss 对 h0 的梯度，Seq2Seq 中用它将 decoder 梯度传给 encoder。
        self.dh_initial = None

    def forward(self, x: np.ndarray, h_prev = None) -> np.ndarray:
        x = np.asanyarray(x)
        self.inputs = x
        
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))
        else:
            h_prev = np.asarray(h_prev).reshape(self.hidden_size, 1)
        # states[0] 保存 h0
        hidden_states = [h_prev.copy()]
        for x_t in x:
            x_t = x_t.reshape(-1, 1)
            h_prev = np.tanh(self.w_h @ h_prev + self.w_x @ x_t + self.b)
            hidden_states.append(h_prev)

        self.hidden_states = np.stack(hidden_states, axis=0)
        # 不把 h0 当成输出
        sequence_output = self.hidden_states[1:, :, 0]  # (T, H)
        if self.return_sequences:
            self.output = sequence_output
        else:
            self.output = sequence_output[-1]
        
        return self.output

    def backward(self, dout:np.ndarray):
        dout = np.asarray(dout)
        sequence_length = self.inputs.shape[0]

        if self.return_sequences:
            expected_shape = (sequence_length, self.hidden_size)
            if dout.shape != expected_shape:
                raise ValueError(
                    f"dout形状应为{expected_shape}，实际为{dout.shape}"
                )
            direct_dh = dout
        else:
            expected_shape = (self.hidden_size,)
            if dout.shape != expected_shape:
                raise ValueError(
                    f"dout形状应为{expected_shape}，实际为{dout.shape}"
                )
            direct_dh = np.zeros(
                (sequence_length, self.hidden_size),
                dtype=np.result_type(dout.dtype, self.inputs.dtype),
            )
            direct_dh[-1] = dout

        gradient_dtype = np.result_type(dout.dtype, self.inputs.dtype)
        dh_next = np.zeros((self.hidden_size, 1), dtype=gradient_dtype)
        d_inputs = np.zeros_like(self.inputs, dtype=gradient_dtype)
        for t in range(sequence_length - 1, -1, -1):
            h_t = self.hidden_states[t + 1]
            h_prev = self.hidden_states[t]

            dh = direct_dh[t].reshape(self.hidden_size, 1) + dh_next

            dz = dh * (1 - h_t ** 2)  # tanh的导数

            self.dw_h += dz @ h_prev.T
            self.dw_x += dz @ self.inputs[t].reshape(1, self.input_size)
            self.db += dz

            dh_next = self.w_h.T @ dz
            d_inputs[t] = (self.w_x.T @ dz).reshape(-1)

        self.dh_initial = dh_next.reshape(-1)
        return d_inputs

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
