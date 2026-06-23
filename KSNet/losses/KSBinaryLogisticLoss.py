import numpy as np
from KSNet.core import KSLossBase

class BinaryLogisticLoss(KSLossBase):
    """
    Logistic 二元逻辑损失（二元交叉熵损失）
    输入：网络原始线性输出 z (未sigmoid)，标签 0/1
    自动内部计算sigmoid，返回平均损失；反向输出 dL/dz
    """
    def __init__(self):
        super().__init__()
        self.sigmoid_out: np.ndarray | None = None  # 缓存sigmoid结果

    def forward(self, pred: np.ndarray, label: np.ndarray) -> float:
        # 维度校验
        if pred.ndim != 2 or label.ndim != 2:
            raise ValueError("pred / label 必须为2维 (batch, 1)")
        if pred.shape != label.shape:
            raise ValueError(f"预测与标签形状不匹配 pred:{pred.shape}, label:{label.shape}")

        self.pred = pred
        self.label = label
        self.batch_size = pred.shape[0]

        # 数值稳定 sigmoid，防止exp溢出
        z = pred
        mask = z >= 0
        self.sigmoid_out = np.empty_like(z)
        self.sigmoid_out[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
        exp_z = np.exp(z[~mask])
        self.sigmoid_out[~mask] = exp_z / (1.0 + exp_z)

        # 裁剪防止 log(0)
        eps = 1e-12
        sig = np.clip(self.sigmoid_out, eps, 1 - eps)

        # 计算单样本损失
        loss_per_sample = -(label * np.log(sig) + (1 - label) * np.log(1 - sig))
        avg_loss = float(np.sum(loss_per_sample) / self.batch_size)
        return avg_loss

    def backward(self) -> np.ndarray:
        """返回 dL/dz，作为最后一层Linear的dout输入"""
        if self.pred is None or self.sigmoid_out is None:
            raise RuntimeError("请先执行 forward 再调用 backward")
        # 化简梯度公式
        dz = (self.sigmoid_out - self.label) / self.batch_size
        return dz