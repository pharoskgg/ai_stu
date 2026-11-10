from numbers import Integral

import numpy as np

from ...tensor import Tensor
from .. import functional
from ..parameter import Parameter
from .module import Module


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

        for name, value in (("num_embeddings", num_embeddings), ("embedding_dim", embedding_dim)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} 必须是整数")
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(np.empty((num_embeddings, embedding_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重。"""
        stdv = 1.0 / np.sqrt(self.embedding_dim)
        self.weight.data[:] = np.random.uniform(-stdv, stdv, size=self.weight.shape)

    def forward(self, input: Tensor) -> Tensor:
        """将每个索引替换为向量，输出形状为 input.shape + (embedding_dim,)。"""
        return functional.embedding(input, self.weight)
