# MiniTorch 职责拆分

原始学习文件 `autograd/02my_stu_autograde.py` 完整保留。新模块迁移其现有实现，仅增加导入和调用连接；数值计算与求导逻辑沿用原代码。

原文件和拆分后的包各自定义自己的类。学习新结构时统一从 `minitorch` 导入，避免混用两套 Tensor。

## 文件职责

```text
minitorch/
├── __init__.py               # 导出 Tensor、nn、optim
├── tensor.py                 # 数据、梯度状态、运算符及 backward 入口
├── nn/
│   ├── __init__.py            # 导出 Parameter、Module、Linear、ReLU、Tanh、Sequential、RNN、functional
│   ├── parameter.py           # 默认需要梯度的叶子 Tensor
│   ├── functional.py          # 无参数网络函数，如 relu
│   └── modules/
│       ├── __init__.py
│       ├── module.py          # 调用入口、参数及子模块注册、递归管理
│       ├── linear.py          # 全连接层，组合矩阵乘法和加法
│       ├── activation.py      # ReLU、Tanh 模块，调用 functional
│       ├── rnn.py             # 单层单向 RNN，返回序列输出和最终状态
│       └── container.py       # Sequential，按注册顺序串联模块
├── optim/
│   ├── __init__.py            # 导出 Optimizer、SGD
│   ├── optimizer.py           # 叶子参数管理、去重和梯度清空
│   └── sgd.py                 # 普通梯度下降，原地更新数据
├── ops/
│   ├── __init__.py            # 导出内置算子
│   ├── arithmetic.py          # Mul、Neg、Add、Pow
│   ├── linalg.py              # MatMul
│   ├── reductions.py          # Sum、Mean
│   ├── shape.py               # Reshape、Transpose、Select、Stack 及维度校验
│   └── activation.py          # ReLU、Tanh 前向及局部反向
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

## Tensor 形状查询与变换

形状接口参考 PyTorch 的 [Tensor.size](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.size.html)、
[Tensor.reshape](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html)
和 [transpose](https://docs.pytorch.org/docs/stable/generated/torch.transpose.html)。

- `x.shape`、`x.size()` 返回形状元组；`x.ndim` 返回维度数量；`x.numel()` 返回元素总数。
- `x.size(dim)` 返回指定维度的长度，支持负维度；标量没有可查询的维度，`size(0)` 也会报错。
- `x.reshape(2, 3)`、`x.reshape((2, 3))`、`x.reshape([2, 3])` 改变形状，保持元素顺序和总数；允许一个 `-1` 自动推导。`reshape(())` 将单元素 Tensor 转为标量，空 Tensor 可变为元素数仍为零的形状。
- `x.transpose(dim0, dim1)` 仅交换两个维度，支持负维度；同一维度交换保持形状。标量允许 `0` 和 `-1`，结果仍为标量。

维度参数必须是整数（可使用 NumPy 整数，不接受布尔值），类型错误抛出
`TypeError`，维度越界抛出 `IndexError`。reshape 的元素数不匹配、多个 `-1`、
小于 `-1` 的维度或空 Tensor 的 `reshape(0, -1)` 等无法确定形状的情况抛出
`RuntimeError`。

调用关系为 `Tensor.reshape/transpose → ops.Reshape/Transpose.apply`。
形状与维度通过关键字配置传入算子，不作为计算图的父 Tensor。
反向时 reshape 恢复输入形状，transpose 再交换同一对维度；共享路径与重复
backward 的梯度累加仍由 engine 处理。常量输入不建图。

```python
from minitorch import Tensor

x = Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
print(x.shape, x.ndim, x.size(-1), x.numel())  # (2, 3) 2 3 6
y = x.transpose(0, 1).reshape(-1)
print(y.data)  # [1. 4. 2. 5. 3. 6.]
(y * Tensor([1., 2., 3., 4., 5., 6.])).sum().backward()
print(x.grad)  # [[1. 3. 5.], [2. 4. 6.]]
```

当前与 PyTorch 的差异：`shape` 和 `size()` 返回普通 tuple，而非 `torch.Size`；
所有结果沿用现有 Tensor 构造时复制 NumPy 数据的行为，修改结果 `.data` 不会
修改输入，因此 transpose 尚不提供 PyTorch 的共享存储视图语义。暂不提供
`view`、`permute`、`squeeze`、`unsqueeze` 或顶层形状函数；矩阵乘法和 Linear
仍沿用原来的维度要求。

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

## ReLU 算子、函数与模块

ReLU 分成三个接口，调用关系为
`nn.ReLU.forward → nn.functional.relu → ops.ReLU.apply`。
`ops.ReLU` 继承 Function，负责 NumPy 前向计算和局部反向公式；
`nn.functional.relu(input)` 提供函数调用；`nn.ReLU()` 继承 Module，
提供可注册为子模块的无参数激活层。

前向逐元素计算 `max(input, 0)`，保留标量及任意批次形状，不修改输入。
反向将上游梯度乘以前向保存的 `input > 0` 掩码，零点梯度取零。
常量输入不建立计算图；共享路径的梯度由现有 engine 累加。
当前接口不提供 inplace 参数或 Tensor.relu 方法。

在 `06autograde/01lesson` 下可组合多层网络：

```python
from minitorch import Tensor, nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.activation = nn.ReLU()
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        return self.output(self.activation(self.hidden(x)))

model = MLP()
x = Tensor([[1., -2.], [-1., 2.]], requires_grad=True)
target = Tensor([[0.], [1.]])
loss = ((model(x) - target) ** 2).mean()
loss.backward()
print(x.grad)
print([name for name, _ in model.named_parameters()])
# ['hidden.weight', 'hidden.bias', 'output.weight', 'output.bias']

print(nn.functional.relu(Tensor([-2., 0., 3.])).data)  # [0., 0., 3.]
```

## Optimizer 与 SGD 训练

通过 `from minitorch import optim` 使用优化器。
`optim.Optimizer(params)` 在构造时保存有序叶子 Tensor 引用，接受
`model.parameters()` 这样的生成器，按对象身份去重；空参数、非 Tensor
和带有 `grad_fn` 的非叶子 Tensor 会被拒绝。基类的 `step()` 抛出
`NotImplementedError`，具体更新规则由子类实现。

`optim.SGD(params, lr=0.01)` 执行普通梯度下降
`parameter.data -= lr * parameter.grad`。学习率必须是有限非负实数，
允许零，拒绝布尔值。更新在原 NumPy 数据数组上完成，保留参数对象和
数据数组身份；共享参数只更新一次。`grad is None` 的参数会跳过，
已有梯度会参与更新，即使参数的 `requires_grad=False`。

`optimizer.zero_grad()` 默认将梯度设为 None；
`optimizer.zero_grad(set_to_none=False)` 将已有梯度数组原地置零，
保留数组身份，也不会为 None 梯度分配数组。
`step()` 不清除梯度，因此每轮训练先清空梯度，再前向、反向并更新。

在 `06autograde/01lesson` 下可运行完整线性回归训练：

```python
import numpy as np
from minitorch import Tensor, nn, optim

np.random.seed(7)
rng = np.random.default_rng(7)
inputs = rng.normal(size=(64, 2))
x = Tensor(inputs)
target = Tensor(inputs @ np.array([[2.0], [-3.0]]) + 0.5)

model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
initial_loss = float(((model(x) - target) ** 2).mean().data)

for _ in range(150):
    optimizer.zero_grad()
    prediction = model(x)
    loss = ((prediction - target) ** 2).mean()
    loss.backward()
    optimizer.step()

final_loss = float(((model(x) - target) ** 2).mean().data)
print("loss:", initial_loss, "->", final_loss)  # 最终损失小于 1e-8
print("weight:", model.weight.data)            # 约为 [[2.], [-3.]]
print("bias:", model.bias.data)                # 约为 [0.5]
```

## Sequential 顺序网络容器

`nn.Sequential(*modules)` 按注册顺序将前一层的输出交给下一层。
位置参数会注册为 `"0"`、`"1"` 等名称；也可传入单个
`collections.OrderedDict` 指定名称和顺序。所有元素必须是 Module，
包括 None 在内的非法元素都会被拒绝；名称沿用 Module 的注册校验。
空容器直接返回输入对象。

容器支持 `len(model)`、`for layer in model` 和整数正负索引
`model[0]`、`model[-1]`。越界抛出 IndexError，非整数（含布尔值）
抛出 TypeError；当前不提供切片、增删索引、append 或 ModuleList。
共享模块在每个注册位置都会执行，递归参数遍历和优化器则默认去重。
参数名称包含容器路径；train、eval 和 zero_grad 沿用 Module 的递归管理。

```python
from collections import OrderedDict
from minitorch import nn

model = nn.Sequential(OrderedDict([
    ("hidden", nn.Linear(2, 8)),
    ("activation", nn.ReLU()),
    ("output", nn.Linear(8, 1)),
]))
print(len(model))                      # 3
print(model[0] is model.hidden)        # True
print(model[-1] is model.output)       # True
print([name for name, _ in model.named_parameters()])
# ['hidden.weight', 'hidden.bias', 'output.weight', 'output.bias']
model.eval()
model.train()
```

在 `06autograde/01lesson` 下运行完整多层网络训练：

```python
import numpy as np
from minitorch import Tensor, nn, optim

np.random.seed(7)
inputs = np.random.default_rng(7).normal(size=(64, 2))
x = Tensor(inputs)
target = Tensor((1.5 * np.maximum(inputs[:, 0], 0)
                 - 0.7 * np.maximum(-inputs[:, 1], 0) + 0.2)[:, None])
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
optimizer = optim.SGD(model.parameters(), lr=0.05)
initial_loss = float(((model(x) - target) ** 2).mean().data)

for _ in range(300):
    optimizer.zero_grad()
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    optimizer.step()

final_loss = float(((model(x) - target) ** 2).mean().data)
print("loss:", initial_loss, "->", final_loss)
# 最终损失低于初始损失的 5%，两个 Linear 的参数都收到有限梯度。
```

在仓库根目录运行测试，使用已安装 NumPy 的 Python：

```bash
python -m unittest discover -s 06autograde/01lesson/tests -v
```

测试覆盖广播、矩阵乘法、共享节点、梯度累加与清空、长计算链、错误检查、包导入、归约运算，以及网络参数注册、共享对象去重、模式切换、完整前向反向计算、ReLU 零点梯度和多层网络梯度、优化器原地更新和线性回归收敛、Sequential 有序执行及多层网络训练收敛。

## Tanh 与单层 RNN

`nn.Tanh()` 调用 `nn.functional.tanh(input)`，再由 `ops.Tanh` 执行 NumPy
前向和局部反向。反向使用前向保存的结果计算 `grad_output * (1 - output**2)`。

`nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity="tanh", bias=True,
batch_first=False)` 对齐 PyTorch 的单层单向基本接口：

- 默认输入为 `(seq_len, batch, input_size)`；`batch_first=True` 时为
  `(batch, seq_len, input_size)`。无批次输入为 `(seq_len, input_size)`，此时忽略 `batch_first`。
- 调用 `output, h_n = model(input, hx=None)`。`output` 包含每个时间步的隐藏状态，
  维度顺序与输入一致，最后一维为 `hidden_size`。
- 有批次时，`hx` 和 `h_n` 均为 `(1, batch, hidden_size)`；无批次时均为
  `(1, hidden_size)`。`batch_first` 不改变隐藏状态的维度顺序。省略 `hx` 使用零状态。
- 参数为 `weight_ih_l0`、`weight_hh_l0`、`bias_ih_l0`、`bias_hh_l0`，权重形状分别为
  `(hidden_size, input_size)` 和 `(hidden_size, hidden_size)`。所有参数均匀初始化到
  `[-1/sqrt(hidden_size), 1/sqrt(hidden_size)]`；`reset_parameters()` 可重新初始化。
- 支持 `nonlinearity="tanh"` 或 `"relu"`，`bias=False` 可关闭两个偏置。
  空序列、特征维度错误和隐藏状态形状不匹配会报错。

当前仅实现 `num_layers=1`，其他层数明确抛出 `NotImplementedError`。暂不支持
双向、dropout、PackedSequence、device/dtype 参数；数值类型沿用 Tensor 的 float64。
这是 PyTorch 接口的教学子集。

```python
import numpy as np
from minitorch import Tensor, nn

model = nn.RNN(input_size=3, hidden_size=5, batch_first=True)
x = Tensor(np.random.randn(2, 4, 3), requires_grad=True)
h0 = Tensor(np.zeros((1, 2, 5)), requires_grad=True)
output, h_n = model(input=x, hx=h0)
print(output.shape)  # (2, 4, 5)
print(h_n.shape)     # (1, 2, 5)
(output ** 2).mean().backward()
print(x.grad.shape)  # (2, 4, 3)
```

RNN 通过可求导的 `Tensor.select(dim, index)` 选取时间步，并用
`minitorch.stack(tensors, dim=0)` 堆叠输出。前者反向将梯度写回原切片，后者将梯度
分给各输入；整个循环保留计算图。参数跨时间步共享和 `output`、`h_n` 的梯度汇合
均由现有 autograd engine 处理。
