from __future__ import annotations

import numpy as np
from KSNet.core.KSLossBase import KSLossBase

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

        eps = 1e-8
        loss_per_sample = -(label * np.log(pred + eps) + (1 - label) * np.log(1 - pred + eps))

        avg_loss = float(np.sum(loss_per_sample) / self.batch_size)
        return avg_loss

    def backward(self) -> np.ndarray:
        """返回 dL/dz，作为最后一层Linear的dout输入"""
        if self.pred is None or self.sigmoid_out is None:
            raise RuntimeError("请先执行 forward 再调用 backward")
        eps = 1e-12
        p = np.clip(self.pred, eps, 1 - eps)
        # dL/dp = (p - y) / (N * p * (1 - p))
        dp = (p - self.label) / (self.batch_size * p * (1 - p))
        self.dz = dp
        return dp
