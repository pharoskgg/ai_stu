import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

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
        self.b_celll = np.zeros(hidden_size)
        
    @staticmethod
    def tanh(x: np.ndarray):
        return np.tanh(x)

    @staticmethod
    def sigmoid(x: np.ndarray):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, inputs: np.ndarray, h=None, c_h=None):
        if h is None:
            h = np.zeros(self.hidden_size)
        if c_h is None:
            c_h = np.zeros(self.hidden_size)

        outputs = []
        for x in inputs:
            f_f = self.sigmoid(x @ self.weight_forget + h @ self.hidden_forget + self.b_forget)
            i_f = self.sigmoid(x @ self.weight_input + h @ self.hidden_input + self.b_input)
            o_f = self.sigmoid(x @ self.weight_output + h @ self.hidden_output + self.b_output)
            mem_candidate = self.tanh(x @ self.weight_cell + h @ self.hidden_cell + self.b_celll)

            c_h = f_f * c_h + i_f * mem_candidate

            h = o_f * self.tanh(c_h)
            outputs.append(h.copy())

        return np.stack(outputs), (h, c_h)

    def backward(self, dout: np.ndarray):

        pass