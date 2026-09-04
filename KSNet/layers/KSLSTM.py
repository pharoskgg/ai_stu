import numpy as np

from KSNet.core.KSangNet import KSNet
from KSNet.layers.KSSigmoid import KSSigmoid


class KSLSTM(KSNet):
    def __init__(self, input_size, hidden_size, return_sequences=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        scale = 0.01

        self.weight_input = np.random.randn(input_size, hidden_size) * scale
        self.hidden_input = np.random.randn(hidden_size, hidden_size) * scale
        self.b_input = np.zeros(hidden_size)

        self.weight_forget = np.random.randn(input_size, hidden_size) * scale
        self.hidden_forget = np.random.randn(hidden_size, hidden_size) * scale
        self.b_forget = np.zeros(hidden_size)

        self.weight_output = np.random.randn(input_size, hidden_size) * scale
        self.hidden_output = np.random.randn(hidden_size, hidden_size) * scale
        self.b_output = np.zeros(hidden_size)

        self.weight_cell = np.random.randn(input_size, hidden_size) * scale
        self.hidden_cell = np.random.randn(hidden_size, hidden_size) * scale
        self.b_cell = np.zeros(hidden_size)

        self.dw_forget = np.zeros_like(self.weight_forget)
        self.du_forget = np.zeros_like(self.hidden_forget)
        self.db_forget = np.zeros_like(self.b_forget)

        self.dw_input = np.zeros_like(self.weight_input)
        self.du_input = np.zeros_like(self.hidden_input)
        self.db_input = np.zeros_like(self.b_input)

        self.dw_output = np.zeros_like(self.weight_output)
        self.du_output = np.zeros_like(self.hidden_output)
        self.db_output = np.zeros_like(self.b_output)

        self.dw_mem_candidate = np.zeros_like(self.weight_cell)
        self.du_mem_candidate = np.zeros_like(self.hidden_cell)
        self.db_mem_candidate = np.zeros_like(self.b_cell)

    @staticmethod
    def tanh(x: np.ndarray):
        return np.tanh(x)

    def forward(self, inputs: np.ndarray, state=None):
        if state is None:
            h = np.zeros(self.hidden_size)
            c_t = np.zeros(self.hidden_size)
        else:
            h, c_t = state

        self.cache = []
        outputs = []
        self.inputs = inputs
        self.input = inputs

        for x in inputs:
            h_prev = h
            c_prev = c_t

            forget_gate = KSSigmoid.apply(x @ self.weight_forget + h_prev @ self.hidden_forget + self.b_forget)
            input_gate = KSSigmoid.apply(x @ self.weight_input + h_prev @ self.hidden_input + self.b_input)
            output_gate = KSSigmoid.apply(x @ self.weight_output + h_prev @ self.hidden_output + self.b_output)
            mem_candidate = self.tanh(x @ self.weight_cell + h_prev @ self.hidden_cell + self.b_cell)

            c_t = (forget_gate * c_prev + input_gate * mem_candidate)

            h = output_gate * self.tanh(c_t)
            outputs.append(h.copy())

            self.cache.append((x, h_prev, c_prev, forget_gate, input_gate, output_gate, mem_candidate, c_t,))

        all_h = np.stack(outputs)
        output = all_h if self.return_sequences else h
        self.output = output

        return output, (h, c_t)

    def backward(self, dout: np.ndarray):
        time_steps = len(self.cache)

        # 将两种输出模式统一转换为每个时间步的梯度
        if self.return_sequences:
            doutputs = dout
        else:
            doutputs = np.zeros((time_steps, self.hidden_size))
            doutputs[-1] = dout

        dh_next = np.zeros(self.hidden_size)
        dc_future = np.zeros(self.hidden_size)
        forget_future = np.zeros(self.hidden_size)
        dx = np.zeros_like(self.inputs)

        for t in reversed(range(time_steps)):
            (x, h_prev, c_prev, forget_gate, input_gate, output_gate, mem_candidate, c_t,) = self.cache[t]
            dh = doutputs[t] + dh_next

            dot = dh * self.tanh(c_t)

            dct = dh * output_gate * (1 - self.tanh(c_t) ** 2) + dc_future * forget_future

            dforget_t = dct * c_prev
            dinput_t = dct * mem_candidate
            dmem_candidate = dct * input_gate

            # dct_prev = dct * forget_gate

            da_f = dforget_t * forget_gate * (1 - forget_gate)
            da_i = dinput_t * input_gate * (1 - input_gate)
            da_o = dot * output_gate * (1 - output_gate)
            da_mem = dmem_candidate * (1 - mem_candidate ** 2)

            if self.trainable:
                self.dw_forget += np.outer(x, da_f)
                self.du_forget += np.outer(h_prev, da_f)
                self.db_forget += da_f

                self.dw_input += np.outer(x, da_i)
                self.du_input += np.outer(h_prev, da_i)
                self.db_input += da_i

                self.dw_output += np.outer(x, da_o)
                self.du_output += np.outer(h_prev, da_o)
                self.db_output += da_o

                self.dw_mem_candidate += np.outer(x, da_mem)
                self.du_mem_candidate += np.outer(h_prev, da_mem)
                self.db_mem_candidate += da_mem

            dc_future = dct
            forget_future = forget_gate

            dh_next = da_f @ self.hidden_forget.T + da_o @ self.hidden_output.T + \
                da_i @ self.hidden_input.T + da_mem @ self.hidden_cell.T

            dx[t] = da_f @ self.weight_forget.T + da_i @ self.weight_input.T + \
                da_mem @ self.weight_cell.T + da_o @ self.weight_output.T

        return dx

    def parameters(self):
        if not self.trainable:
            return []

        return [
            (self.weight_forget, self.dw_forget),
            (self.hidden_forget, self.du_forget),
            (self.b_forget, self.db_forget),
            (self.weight_input, self.dw_input),
            (self.hidden_input, self.du_input),
            (self.b_input, self.db_input),
            (self.weight_output, self.dw_output),
            (self.hidden_output, self.du_output),
            (self.b_output, self.db_output),
            (self.weight_cell, self.dw_mem_candidate),
            (self.hidden_cell, self.du_mem_candidate),
            (self.b_cell, self.db_mem_candidate),
        ]

    def zero_grad(self):
        for _, grad in self.parameters():
            grad.fill(0.0)
