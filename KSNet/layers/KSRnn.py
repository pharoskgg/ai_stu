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

        self.hidden_states = None  # 用于存储每个时间步的隐藏状态，形状 (seq_len, hidden_size, 1)
        self.hidden_state = None  # 用于存储当前时间步的隐藏状态，形状 (hidden_size, 1)

    def forward(self, x: np.ndarray, h_prev = None) -> np.ndarray:
        x = np.asanyarray(x)
        hiddend_states = []
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))
        for x_t in x:
            h_prev = np.tanh(self.w_h @ h_prev + self.w_x @ x_t + self.b)
            hiddend_states.append(h_prev)

        self.hidden_states = np.stack(hiddend_states, axis=0)
        if self.return_sequences:
            self.output = self.hidden_states
        else:
            self.output = self.hidden_states[-1]
        
        return self.output

    def backward(self, dout:np.ndarray):
        dout = np.asarray(dout)
        sequence_length = self.hidden_states.shape[0]

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
                dtype=np.result_type(dout, self.weight),
            )
            direct_dh[-1] = dout

        dh_next = np.zeros(self.hidden_size)
        d_inputs = np.zeros(self.inputs.shape)
        for t in range(sequence_length - 1, -1, -1):
            h_t = self.hidden_states[t + 1]
            h_prev = self.hidden_states[t]

            dh = direct_dh[t] + dh_next

            dz = dh * (1 - h_t ** 2)  # tanh的导数

            self.dw_h += np.outer(dz, h_prev)
            self.dw_x += np.outer(dz, self.inputs[t])
            self.db += dz

            dh_next = self.w_h.T @ dz
            d_inputs[t] = self.w_x.T @ dz

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