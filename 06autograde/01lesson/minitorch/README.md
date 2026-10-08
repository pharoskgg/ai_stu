# MiniTorch 职责拆分

原始学习文件 `autograd/02my_stu_autograde.py` 完整保留。新模块迁移其现有实现，仅增加导入和调用连接；数值计算与求导逻辑沿用原代码。

原文件和拆分后的包各自定义自己的类。学习新结构时统一从 `minitorch` 导入，避免混用两套 Tensor。

## 文件职责

```text
minitorch/
├── __init__.py               # 导出 Tensor
├── tensor.py                 # 数据、梯度状态、运算符及 backward 入口
├── ops/
│   ├── __init__.py            # 导出内置算子
│   ├── arithmetic.py          # Mul、Neg、Add、Pow
│   └── linalg.py              # MatMul
└── autograd/
    ├── __init__.py            # 公开 backward、Context、Function
    ├── function.py            # Context 和 Function.apply：前向执行与建图
    ├── graph.py               # FunctionNode：一次运算的上下文和父 Tensor
    ├── engine.py              # 拓扑排序、反向调度、校验与梯度累加
    ├── utils.py               # sum_to_shape：广播梯度还原
    └── 02my_stu_autograde.py   # 未修改的原始学习代码
```

`Function` 定义一种运算的执行方式；`FunctionNode` 表示该运算的一次调用，保存该次调用的 `Context` 和 `parents`。算子计算局部梯度，engine 组织整张图的反向传播。

## 调用过程

```text
前向：Tensor.__mul__ → Mul.apply → Mul.forward
                              → FunctionNode → 输出 Tensor

反向：Tensor.backward → autograd.backward → engine.run_backward
                                             → FunctionNode.backward
                                             → Mul.backward
                                             → 汇总梯度、写入叶子 .grad
```

两个拓扑排序方法都迁移到 engine；执行时使用原来的栈版本，递归版本保留供学习对照。engine 函数的参数 `self` 沿用原代码命名，表示反向传播的输出 Tensor。

`Function.apply` 内部延迟导入 Tensor，避免 `Tensor → 算子 → Function → Tensor` 的循环导入。`graph.py` 的类型依赖使用 `TYPE_CHECKING`。

本次保留 `parents` 连接方式和 NumPy 实现；Node/Edge、AccumulateGrad 及数值后端拆分留到后续阶段。

## 使用与验证

在仓库根目录执行：

```bash
cd 06autograde/01lesson
python - <<'PY'
from minitorch import Tensor

x = Tensor(3.0, requires_grad=True)
y = x * x + 2 * x
y.backward()
print(y.data)  # 15.0
print(x.grad)  # 8.0
PY
```

也可以调用公开接口 `from minitorch.autograd import backward`，或继承 `Function` 添加自定义算子。

在仓库根目录运行测试，使用已安装 NumPy 的 Python：

```bash
python -m unittest discover -s 06autograde/01lesson/tests -v
```

测试覆盖广播、矩阵乘法、共享节点、梯度累加与清空、长计算链、错误检查以及包导入。
