"""损失模块与 functional 共用的归约参数解析。"""

import warnings


def resolve_reduction(reduction="mean", size_average=None, reduce=None):
    """兼容旧参数优先级，并返回 none、mean 或 sum。"""
    if size_average is not None or reduce is not None:
        for name, value in (("size_average", size_average), ("reduce", reduce)):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} 必须是 bool 或 None")
        average = True if size_average is None else size_average
        aggregate = True if reduce is None else reduce
        reduction = ("mean" if average else "sum") if aggregate else "none"
        warnings.warn(
            f"size_average 和 reduce 已弃用，请改用 reduction={reduction!r}",
            UserWarning, stacklevel=3,
        )
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction 必须为 'none'、'mean' 或 'sum'")
    return reduction
