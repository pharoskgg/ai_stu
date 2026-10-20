"""数值稳定的交叉熵前向计算和局部反向公式。"""

from numbers import Integral, Real

import numpy as np

from ..autograd.function import Context, Function


class CrossEntropy(Function):
    @staticmethod
    def forward(ctx: Context, input, target, weight=None, *, ignore_index=-100,
                reduction="mean", label_smoothing=0.0):
        if input.ndim == 0:
            raise ValueError("交叉熵 input 必须至少为一维")
        class_dim = 0 if input.ndim == 1 else 1
        classes = input.shape[class_dim]
        if classes == 0:
            raise ValueError("交叉熵类别数必须大于 0")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction 必须为 'none'、'mean' 或 'sum'")
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
            raise TypeError("ignore_index 必须是整数")
        if isinstance(label_smoothing, bool) or not isinstance(label_smoothing, Real):
            raise TypeError("label_smoothing 必须是实数")
        if not np.isfinite(label_smoothing) or not 0 <= label_smoothing <= 1:
            raise ValueError("label_smoothing 必须在 [0, 1] 范围内")
        if weight is not None and weight.shape != (classes,):
            raise ValueError(f"weight 的形状必须为 ({classes},)")

        moved = np.moveaxis(input, class_dim, -1)
        output_shape = moved.shape[:-1]
        rows = moved.reshape(-1, classes)
        soft_target = target.shape == input.shape
        weights = np.ones(classes, dtype=input.dtype) if weight is None else weight
        valid = np.ones(rows.shape[0], dtype=bool)

        if soft_target:
            if ignore_index >= 0:
                raise ValueError("概率目标不支持 ignore_index")
            distribution = np.moveaxis(target, class_dim, -1).reshape(-1, classes)
            normalizer = rows.shape[0]
        else:
            if target.shape != output_shape:
                raise ValueError(f"类别索引 target 的形状必须为 {output_shape}，实际为 {target.shape}")
            if ctx.needs_input_grad[1]:
                raise RuntimeError("类别索引 target 不支持梯度")
            if weight is not None and ctx.needs_input_grad[2]:
                raise RuntimeError("类别索引目标的 weight 不支持梯度")
            labels = target.reshape(-1)
            if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
                raise ValueError("类别索引 target 必须是有限整数值")
            valid = labels != ignore_index
            if np.any((labels[valid] < 0) | (labels[valid] >= classes)):
                raise IndexError(f"类别索引必须在 [0, {classes}) 范围内")
            # 先屏蔽忽略位置，再转整数，避免越界或溢出索引。
            indices = np.where(valid, labels, 0).astype(np.intp)
            distribution = np.zeros_like(rows)
            distribution[np.arange(rows.shape[0]), indices] = 1
            normalizer = weights[indices[valid]].sum()

        distribution = (1 - label_smoothing) * distribution + label_smoothing / classes
        if not soft_target:
            distribution[~valid] = 0

        # 先减去各行最大值，再计算 log-softmax，避免 exp(logits) 溢出。
        shifted = rows - rows.max(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        if not soft_target and label_smoothing == 0:
            # 索引目标只读取选中类别，避免未选中类别的 0 * (-Inf) 污染损失。
            losses = np.zeros(rows.shape[0], dtype=input.dtype)
            losses[valid] = -weights[indices[valid]] * log_probabilities[
                np.flatnonzero(valid), indices[valid]
            ]
        else:
            losses = -(distribution * weights * log_probabilities).sum(axis=1)
            if not soft_target:
                losses[~valid] = 0
        ctx.save_for_backward(log_probabilities, distribution, weights, valid)
        ctx.moved_shape = moved.shape
        ctx.class_dim = class_dim
        ctx.soft_target = soft_target
        ctx.reduction = reduction
        ctx.normalizer = normalizer
        ctx.label_smoothing = label_smoothing

        if reduction == "none":
            return losses.reshape(output_shape)
        if reduction == "sum":
            return losses.sum()
        # 与 PyTorch 一致，空输入、全忽略或零总权重的 mean 可以为 NaN。
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(losses.sum(), normalizer)

    @staticmethod
    def backward(ctx: Context, grad_output):
        log_probabilities, distribution, weights, valid = ctx.saved_values
        scale = grad_output.reshape(-1, 1) if ctx.reduction == "none" else grad_output
        grad_input = grad_target = grad_weight = None

        def restore_shape(gradient):
            return np.moveaxis(gradient.reshape(ctx.moved_shape), -1, ctx.class_dim)

        with np.errstate(divide="ignore", invalid="ignore"):
            if ctx.reduction == "mean":
                scale = np.divide(scale, ctx.normalizer)
            if ctx.needs_input_grad[0]:
                if not ctx.soft_target and ctx.label_smoothing == 0:
                    # 无平滑索引目标只使用选中权重，跳过其他类别的 Inf/NaN。
                    weighted = np.zeros_like(distribution)
                    np.multiply(distribution, weights, out=weighted, where=distribution != 0)
                else:
                    weighted = distribution * weights
                gradient = (np.exp(log_probabilities) * weighted.sum(axis=1, keepdims=True)
                            - weighted) * scale
                if not ctx.soft_target:
                    gradient[~valid] = 0
                grad_input = restore_shape(gradient)
            if ctx.soft_target and ctx.needs_input_grad[1]:
                gradient = -log_probabilities * weights * (1 - ctx.label_smoothing) * scale
                grad_target = restore_shape(gradient)
            if len(ctx.needs_input_grad) == 3 and ctx.needs_input_grad[2]:
                grad_weight = (-log_probabilities * distribution * scale).sum(axis=0)

        gradients = (grad_input, grad_target)
        return gradients + (grad_weight,) if len(ctx.needs_input_grad) == 3 else gradients
