import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, return_sequences=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        self.weight = np.random.randn(input_size, hidden_size) * 0.1
        self.w_hh = np.random.randn(hidden_size, hidden_size) * 0.1

        self.bias = np.zeros(self.hidden_size)
        self.input = None
        self.hidden_states = None
        self.output = None

    def forward(self, inputs):
        inputs = np.asarray(inputs)
        if inputs.ndim != 2:
            raise ValueError(
                f"RNN输入必须是(sequence_length, input_size)，实际为{inputs.shape}"
            )
        if inputs.shape[1] != self.input_size:
            raise ValueError(
                f"输入特征维度应为{self.input_size}，实际为{inputs.shape[1]}"
            )
        if inputs.shape[0] == 0:
            raise ValueError("输入序列不能为空")

        self.input = inputs
        hidden_states = [np.zeros(self.hidden_size)]
        for x_t in inputs:
            z = (
                x_t @ self.weight
                + hidden_states[-1] @ self.w_hh
                + self.bias
            )
            hidden_states.append(np.tanh(z))

        # h0也要缓存供BPTT使用，但不能作为输入序列产生的输出交给下一层。
        self.hidden_states = np.stack(hidden_states, axis=0)
        if self.return_sequences:
            self.output = self.hidden_states[1:]
        else:
            self.output = self.hidden_states[-1]
        return self.output

    def backward(self, dout):
        
