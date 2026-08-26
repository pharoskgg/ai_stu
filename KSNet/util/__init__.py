from .util_make_datasets import generate_moons_dataset, generate_multiclass_dataset

__all__ = [
    "generate_moons_dataset",
    "generate_multiclass_dataset",
    "plot_training_history",
    "setup_chinese_font",
]


def __getattr__(name):
    """按需导入绘图工具，纯数据实验无需提前加载 matplotlib。"""
    if name in {"plot_training_history", "setup_chinese_font"}:
        from .util_plot import plot_training_history, setup_chinese_font

        globals().update(
            plot_training_history=plot_training_history,
            setup_chinese_font=setup_chinese_font,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
