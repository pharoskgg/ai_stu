# MiniTorch 职责拆分

原始学习文件 `autograd/02my_stu_autograde.py` 完整保留。新模块迁移其现有实现，仅增加导入和调用连接；数值计算与求导逻辑沿用原代码。

原文件和拆分后的包各自定义自己的类。学习新结构时统一从 `minitorch` 导入，避免混用两套 Tensor。

## 文件职责

```text
minitorch/
├── __init__.py               # 导出 Tensor、nn
├── tensor.py                 # 数据、梯度状态、运算符及 backward 入口
├── nn/
│   ├── __init__.py            # 导出 Parameter、Module、Linear
│   ├── parameter.py           # 默认需要梯度的叶子 Tensor
│   └── modules/
│       ├── __init__.py
│       ├── module.py          # 调用入口、参数及子模块注册、递归管理
│       └── linear.py          # 全连接层，组合矩阵乘法和加法
├── ops/
│   ├── __init__.py            # 导出内置算子
│   ├── arithmetic.py          # Mul、Neg、Add、Pow
│   ├── linalg.py              # MatMul
│   └── reductions.py          # Sum、Mean
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

`Tensor.sum(dim=None, keepdim=False)` 和 `Tensor.mean(dim=None, keepdim=False)` 对应归约算子。`Function.apply` 将 `axis`、`keepdims` 等关键字配置传给前向计算，不将它们加入计算图输入。

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

可以使用归约算子组合均方误差损失：

```python
loss = ((prediction - target) ** 2).mean()
loss.backward()
```

迁移保留原实现的边界行为：标量的默认 `mean()` 正常，但 `mean(dim=0)` 和 `mean(dim=-1)` 会由 NumPy 抛出轴错误。

## Parameter 与 Module

可以通过 `from minitorch import nn` 或 `from minitorch.nn import Parameter, Module` 使用网络接口。目录边界和核心参数管理接口参考 [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)。

`Parameter` 继承 Tensor，默认 `requires_grad=True`，所以现有算子和 engine 可直接计算其梯度。构造时沿用当前 Tensor 的 float64 和数据复制行为；传入 Tensor 时复制其数据，创建独立叶子，不继承原计算图或梯度。可用 `requires_grad=False` 冻结参数，冻结参数仍在 `parameters()` 中。

`Module` 提供这些接口：

- 重写 `forward`，通过 `model(x, **kwargs)` 调用。
- 直接赋值 `self.weight = Parameter(...)`、`self.layer = Module(...)` 自动注册；子类须先调用 `super().__init__()`。
- `register_parameter(name, param)`、`add_module(name, module)` 显式注册，支持 None 占位。
- `parameters()`、`named_parameters(prefix="", recurse=True, remove_duplicate=True)` 递归遍历，默认去重；`recurse=False` 仅返回当前模块参数。
- `children()`、`named_children()` 遍历直接子模块；`modules()`、`named_modules()` 遍历包含自身的所有模块，默认去重。
- `train()`、`eval()` 递归切换 `training` 标记，不改变参数的梯度开关。
- `zero_grad()` 将注册参数的梯度设为 None；`zero_grad(set_to_none=False)` 将已有梯度数组原地置零。

普通 Tensor 和放在普通 list/dict 中的对象不会自动注册。已注册参数只能被 Parameter 或 None 替换；已注册子模块可以替换为 Module、Parameter 或 None，删除属性会取消注册。名称不能为空、包含点号或覆盖 Module 接口；模块间的循环包含关系会被拒绝。

在 `06autograde/01lesson` 下可运行以下示例，自定义一个全连接计算模块：

```python
from minitorch import Tensor, nn

class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter([[1.0], [2.0]])
        self.bias = nn.Parameter([0.5])

    def forward(self, x):
        return x @ self.weight + self.bias

model = Affine()
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
target = Tensor([[0.0], [0.0]])

model.zero_grad()
loss = ((model(x) - target) ** 2).mean()
loss.backward()

for name, parameter in model.named_parameters():
    print(name, parameter.grad)  # weight: [[40.], [57.]]；bias: [17.]
    if parameter.grad is not None:
        parameter.data -= 0.001 * parameter.grad
```

示例通过数据数组更新参数；网络模块的反向传播由现有算子组合完成，无需单独定义 Module.backward。

## Linear 全连接层

`nn.Linear(in_features, out_features, bias=True)` 计算 `x @ weight + bias`。
输入必须是至少二维的 Tensor，形状为 `(..., in_features)`；输出形状为
`(..., out_features)`，保留所有批次维度。特征数必须是正整数。

权重形状为 `(in_features, out_features)`，偏置形状为 `(out_features,)`。
两者均在 `[-1 / sqrt(in_features), 1 / sqrt(in_features)]` 中均匀随机初始化，
作为 Parameter 自动注册。设置 `bias=False` 时 `layer.bias` 为 None，
参数遍历只返回权重。可在构造层之前用 `np.random.seed(...)` 固定随机种子。

在 `06autograde/01lesson` 下运行：

```python
from minitorch import Tensor, nn

layer = nn.Linear(2, 1)
layer.weight.data[:] = [[1.0], [2.0]]
layer.bias.data[:] = [0.5]
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

y = layer(x)
print(y.data)  # [[5.5], [11.5]]
(y ** 2).mean().backward()
print(layer.weight.grad)  # [[40.], [57.]]
print(layer.bias.grad)    # [17.]
print(x.grad)             # [[5.5, 11.], [11.5, 23.]]
```

前向传播使用现有 MatMul、Add 算子，输入、权重和偏置的梯度由 autograd
自动计算；高维批次中的权重和偏置梯度会汇总到原参数形状。

在仓库根目录运行测试，使用已安装 NumPy 的 Python：

```bash
python -m unittest discover -s 06autograde/01lesson/tests -v
```

测试覆盖广播、矩阵乘法、共享节点、梯度累加与清空、长计算链、错误检查、包导入、归约运算，以及网络参数注册、共享对象去重、模式切换和完整前向反向计算。
