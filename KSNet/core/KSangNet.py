from abc import ABC, abstractmethod
import numpy as np

class KSNet(ABC):
    """
    深度学习层抽象基类
    所有算子层必须继承该类，实现 forward / backward 两个核心方法
    """
    def __init__(self):
        # ========== 前向传播缓存 ==========
        self.input: np.ndarray | None = None   # 前向输入，形状 (N, ...)
        self.output: np.ndarray | None = None  # 前向输出，形状 (N, ...)

        # ========== 可训练参数 ==========
        self.weight: np.ndarray | None = None  # 权重参数，无参层为 None
        self.bias: np.ndarray | None = None    # 偏置参数，无参层为 None

        # ========== 梯度缓存 ==========
        self.w_grad: np.ndarray | None = None  # 权重梯度，与 weight 同形状
        self.b_grad: np.ndarray | None = None  # 偏置梯度，与 bias 同形状

        # ========== 标志位 ==========
        self.trainable: bool = True  # 是否参与参数更新
        self.training: bool = True   # True=训练模式(计算梯度)，False=推理模式

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        :param x: 输入张量，形状 (batch_size, 输入维度...)
        :return: 输出张量，形状 (batch_size, 输出维度...)
        内部必须缓存 self.input = x，self.output = 计算结果
        """
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播（链式求导）
        :param dout: 上游传来的梯度，形状与 self.output 完全一致
        :return: dx 传递给上一层的梯度，形状与 self.input 完全一致
        有参层内部必须计算并缓存 self.w_grad、self.b_grad
        """
        pass

    def zero_grad(self) -> None:
        """清空梯度，每个batch训练前调用，防止梯度累加"""
        if self.trainable:
            if self.w_grad is not None:
                self.w_grad.fill(0.0)
            if self.b_grad is not None:
                self.b_grad.fill(0.0)

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """
        获取所有可训练参数与对应梯度，供优化器统一调用
        :return: [(参数, 梯度), (参数, 梯度)]
        """
        params = []
        if self.trainable:
            if self.weight is not None:
                params.append((self.weight, self.w_grad))
            if self.bias is not None:
                params.append((self.bias, self.b_grad))
        return params

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def save_weights(self, path: str) -> None:
        """保存权重至npz文件"""
        save_dict = {}
        if self.weight is not None:
            save_dict["weight"] = self.weight
        if self.bias is not None:
            save_dict["bias"] = self.bias
        np.savez(path, **save_dict)

    def load_weights(self, path: str) -> None:
        """从npz加载权重"""
        data = np.load(path)
        if "weight" in data and self.weight is not None:
            self.weight = data["weight"]
        if "bias" in data and self.bias is not None:
            self.bias = data["bias"]