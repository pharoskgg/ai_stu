"""损失模块基类和交叉熵；具体计算由 functional 与算子负责。"""

from .. import functional
from .._reduction import resolve_reduction
from .module import Module


class _Loss(Module):
    """统一保存损失归约方式的内部基类。"""

    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super().__init__()
        self.reduction = resolve_reduction(reduction, size_average, reduce)


class _WeightedLoss(_Loss):
    """将类别权重注册为 buffer，避免被当作模型参数更新。"""

    def __init__(self, weight=None, size_average=None, reduce=None, reduction="mean"):
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)


class CrossEntropyLoss(_WeightedLoss):
    """从 logits 计算交叉熵，支持类别索引和同形状的概率目标。"""

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction="mean", label_smoothing=0.0):
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        return functional.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index,
            reduction=self.reduction, label_smoothing=self.label_smoothing,
        )
