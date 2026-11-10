import numpy as np


DEFAULT_FLOAT_DTYPE = np.dtype(np.float32)

SUPPORTED_DTYPES = frozenset(
    np.dtype(t)
    for t in (
        np.bool_,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    )
)

def normalize_dtype(dtype):
    """将 dtype 转换为 numpy.dtype 对象，并检查是否受支持。"""
    dtype = np.dtype(dtype)
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"不支持的数据类型: {dtype}")

    return dtype

def promote_dtype(left, right):
    """用于合并构造数据中的元素类型。"""
    left = normalize_dtype(left)
    right = normalize_dtype(right)

    floats = [
        dtype
        for dtype in (left, right)
        if dtype.kind == "f"
    ]

    if floats:
        # 整数与浮点混合时，由浮点输入决定浮点精度。
        return max(floats, key=lambda dtype: dtype.itemsize)

    return np.promote_types(left, right)


def infer_dtype(data):
    """推断构造数据的 dtype，序列递归提升，空序列使用默认浮点类型。

    NumPy 数组和标量保留受支持的 dtype；Python bool、int、float
    分别使用 bool、int64 和 DEFAULT_FLOAT_DTYPE。
    """
    if isinstance(data, (np.ndarray, np.generic)):
        return normalize_dtype(data.dtype)

    # bool 是 int 的子类，必须先判断。
    if isinstance(data, bool):
        return np.dtype(np.bool_)
    if isinstance(data, int):
        return np.dtype(np.int64)
    if isinstance(data, float):
        return DEFAULT_FLOAT_DTYPE

    if isinstance(data, (list, tuple)):
        dtype = None
        for item in data:
            item_dtype = infer_dtype(item)
            dtype = (item_dtype if dtype is None
                     else promote_dtype(dtype, item_dtype))
        return DEFAULT_FLOAT_DTYPE if dtype is None else dtype

    raise TypeError(f"无法推断数据类型: {type(data).__name__}")
