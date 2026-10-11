# MiniTorch 训练接口实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按顺序实现最小优化器、ReLU 和 Sequential，每一步独立验证、提交并推送。

**Architecture:** 优化器管理参数引用及更新；ReLU 按算子、functional、Module 分层；Sequential 复用 Module 的子模块注册及递归管理。

**Tech Stack:** Python、NumPy、unittest。

**Spec:** 本次对话已确认的第 1、2、3 点及用户明确授权：按顺序由子代理完成，每点使用两句中文提交信息并同步远程。

## Global Constraints

- 仅修改 minitorch、相邻 tests 和本计划；保留原始 autograd/02my_stu_autograde.py。
- 保持 NumPy float64、现有 Tensor/Module 行为，无新依赖。
- 不实现 momentum、参数组、MSELoss、no_grad 或修改 Linear 的权重布局。
- 三个实现串行运行；实现代理不提交、不推送、不派生代理，由主代理审查后逐步提交并推送 origin/main。
- 各任务补中文 README 用法；用 unittest 做先失败后通过的行为测试，报告红绿验证命令及输出。

## Review Focus

- 参数生成器需物化并保留引用，多次 step 均更新原参数。
- grad=None 与零梯度有区别；step 不清空梯度，不替换 Parameter。
- ReLU 对零的梯度取零，常量输入不建图，共享路径正确累加。
- Sequential 的参数、共享模块去重、train/eval 由既有注册机制统一管理。
- 集成训练通过组合现有算子建图，各层参数都收到有限梯度。

### Task 1: Optimizer 与 SGD

**Files:** 创建 minitorch/optim/{__init__.py,optimizer.py,sgd.py}、tests/test_optim.py；修改 minitorch/__init__.py、minitorch/README.md。（路径相对于 06autograde/01lesson/。）

**Interfaces:** Optimizer(params) 保存有序 Tensor 叶子引用；zero_grad(set_to_none=True)；step() 基类抛 NotImplementedError。SGD(params, lr=0.01) 实现普通梯度下降。公开 from minitorch import optim。

- [ ] 编写失败测试：参数 [1,2] 梯度 [2,-4]、lr=0.1 后得到 [0.8,2.4]；生成器连续两步更新；None 跳过；零梯度；step 不清梯度；zero_grad 两种模式及数组身份；参数身份不变；共享引用仅更新一次；拒绝空参数、非 Tensor、非叶子；lr 拒绝 bool、非实数、负值及非有限值，允许零。
- [ ] 运行新测试确认缺少 optim 而失败。
- [ ] 实现：构造时按对象身份去重并物化参数；按 grad 是否为 None 决定更新（不要以 requires_grad 额外过滤已有梯度）；通过 NumPy 数组原地减法更新；不支持额外优化器选项。
- [ ] 添加固定随机种子线性回归收敛测试及 README 完整训练示例；全套 unittest 和 git diff --check 通过。
- [ ] 主代理独立审查、提交并推送。提交信息两句：实现优化器基类和随机梯度下降，支持参数更新与梯度清空。补充优化器行为测试和线性回归训练示例。

### Task 2: ReLU 算子与网络接口

**Files:** 创建 minitorch/ops/activation.py、minitorch/nn/functional.py、minitorch/nn/modules/activation.py、tests/test_activation.py；修改 ops/__init__.py、nn/{__init__.py,modules/__init__.py}、README.md。

**Interfaces:** ops.ReLU(Function)；nn.functional.relu(input: Tensor) -> Tensor；nn.ReLU(Module).forward(input) -> Tensor。沿用 Function.apply，模块调用 functional，functional 调用算子。无 inplace 参数；本阶段无需 Tensor.relu。

- [ ] 编写失败测试：输入 [-2,0,3] 前向 [0,0,3]，上游 [2,3,4] 梯度 [0,0,4]；标量与批次形状；输入不被修改；常量不建图；共享 ReLU 路径梯度累加；Module 无参数；functional 和 Module 数值一致。
- [ ] 运行新测试确认缺失接口而失败。
- [ ] 实现 np.maximum 与保存前向正值掩码，backward 为上游梯度乘掩码；在零点取零。
- [ ] 补充 Linear → ReLU → Linear 集成反向测试及 README 分层说明；全套 unittest 和 git diff --check 通过。
- [ ] 主代理独立审查、提交并推送。提交信息两句：实现分层的线性整流算子、函数接口和网络模块。补充前向反向传播测试及多层网络用法。

### Task 3: Sequential 网络容器

**Files:** 创建 minitorch/nn/modules/container.py、tests/test_sequential.py；修改 nn/{__init__.py,modules/__init__.py}、README.md。

**Interfaces:** nn.Sequential(*modules)，支持位置 Module 及单个 collections.OrderedDict 命名模块；forward(input) 按注册顺序串联；支持 len、迭代、整数索引（含负数）。本阶段不实现切片、增删索引、append、ModuleList。

- [ ] 编写失败测试：位置层命名为 "0"/"1"；OrderedDict 保持名称和顺序；空容器返回输入本身；非法元素（含 None）拒绝；整数正负索引及越界；非整数索引拒绝；递归参数命名、共享模块重复执行但参数去重、train/eval/zero_grad 传播。
- [ ] 运行新测试确认缺少 Sequential 而失败。
- [ ] 实现复用 add_module 与 _modules 的有序值；forward 必须遍历全部注册位置而非会去重的 children()；不单独保存第二份层列表。
- [ ] 添加 Sequential(Linear, ReLU, Linear) + SGD 的固定数据训练测试（固定随机种子，损失明显下降，各层收到有限梯度）；README 完整用法；全套 unittest 和 git diff --check 通过。
- [ ] 主代理完成独立任务审查及整体审查、提交并推送。提交信息两句：实现顺序网络容器，复用模块注册和递归参数管理。补充容器行为测试及多层网络训练示例。
