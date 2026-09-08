from KSNet.core.KSangNet import KSNet
import numpy as np

class KSEmbedding(KSNet):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.trainable = True

        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.dembedding_matrix = np.zeros_like(self.embedding_matrix)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim not in (1, 2):
            raise ValueError(
                "Embedding 输入必须是一维或二维索引数组，"
                f"实际形状为 {x.shape}"
            )
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError(f"Embedding 输入必须是整数索引，实际 dtype 为 {x.dtype}")
        if np.any((x < 0) | (x >= self.vocab_size)):
            raise ValueError(
                f"输入索引应在 [0, {self.vocab_size}) 范围内，实际为 {x}"
            )

        self.input = x
        self.output = self.embedding_matrix[x]
        return self.output

    def backward(self, dout: np.ndarray):
        dout = np.asarray(dout)

        expected_shape = self.input.shape + (self.embedding_dim,)
        if dout.shape != expected_shape:
            raise ValueError(
                f"dout 形状必须与 Embedding 输出一致，应为 {expected_shape}，"
                f"实际为 {dout.shape}"
            )
        # for index, gradient in zip(indices, values):
        #     target[index] += gradient
        if self.trainable:
            np.add.at(self.dembedding_matrix, self.input.reshape(-1), dout.reshape(-1, self.embedding_dim))
        
        # token 索引是离散整数，不存在有意义的输入梯度。
        # 为兼容 KSNet.backward() 的 ndarray 返回约定，返回同形状零数组。
        return np.zeros(self.input.shape, dtype=dout.dtype)

    def parameters(self):
        if not self.trainable:
            return []
        return [(self.embedding_matrix, self.dembedding_matrix)]
    def zero_grad(self) -> None:
        self.dembedding_matrix.fill(0.0)

    def save_weights(self, path: str) -> None:
        np.savez(path, embedding_matrix=self.embedding_matrix)

    def load_weights(self, path: str) -> None:
        with np.load(path) as data:
            matrix = data["embedding_matrix"]
            if matrix.shape != self.embedding_matrix.shape:
                raise ValueError(...)
            self.embedding_matrix[...] = matrix
