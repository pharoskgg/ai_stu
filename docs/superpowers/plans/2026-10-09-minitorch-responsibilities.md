# MiniTorch 职责拆分实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将单文件自动求导实现拆成职责清晰的 Python 包，保持现有数值行为。

**Architecture:** Tensor 只保存状态并转发运算及 backward；Function.apply 执行前向并建图；FunctionNode 保存每次运算的上下文和父 Tensor；engine 负责遍历、校验和反向执行。算子按算术和矩阵运算分组。

**Tech Stack:** Python、NumPy、unittest。

**Spec:** 本次对话中用户确认的第一步职责拆分方案，具体边界记录如下。

## Global Constraints

- 保留 float64、NumPy 数组复制、广播求和、叶子梯度累加和现有错误检查。
- 保留 parents 图连接；本次不引入 Node/Edge/AccumulateGrad 或数值后端。
- 用户补充要求：原学习文件完整保留，不修改；新模块迁移原实现，只添加必要导入和调用连接。
- 在当前工作区完成用户授权的修改；不修改已有 .gitignore 变更，不提交或推送。

## Review Focus

- 多条路径到同一个 Tensor 时汇总全部梯度，含同一算子的重复输入。
- 广播梯度与批量矩阵乘法梯度必须恢复原输入形状。
- backward 重复调用累加叶子梯度；zero_grad 后重新累加。
- 独立导入各模块及从其他工作目录直接执行旧入口不发生循环导入。
- 保留输入梯度形状及算子返回梯度数量、形状的错误检查。

### Task 1: 包拆分与兼容入口

**Files:** 新建 `06autograde/01lesson/minitorch/{__init__.py,tensor.py,README.md,ops/{__init__.py,arithmetic.py,linalg.py},autograd/{__init__.py,function.py,graph.py,engine.py,utils.py}}`；原学习文件保持不变；测试位于 `06autograde/01lesson/tests/test_autograd.py`。

**Interfaces:** `Tensor.backward(gradient=None)` → `autograd.backward(tensor, gradient=None)` → `engine.run_backward(self, gradient=None)`（`self` 为输出 Tensor，沿用原参数名）；算子继承 `Function` 并实现 `forward/backward`；`FunctionNode(function, ctx, parents)`。

- [x] 编写行为回归测试，在旧实现上确认预期；新包测试因缺少公开接口失败。
- [x] 提取工具、上下文、建图、节点、算子、Tensor 和 engine，处理循环导入。
- [x] 保留旧文件；编写中文职责说明及运行示例。
- [x] 运行 `python -m unittest discover -s 06autograde/01lesson/tests -v`，18 项全部通过。
- [x] 审查迁移前后数值算法、导入及文档；独立代码审查无问题，`git diff --check` 通过。

## 验证记录

- 原文件与 HEAD 中的版本逐字节一致。
- 原实现的 13 项行为测试通过；拆分包的 18 项回归与集成测试通过。
- AST 对比确认工具、Context、FunctionNode、所有算子、Tensor 原有数据及运算方法、两种拓扑排序均未改变。
- Function 仅新增延迟导入；原 backward 迁到 engine 时只调整函数名称和拓扑函数调用连接，Tensor.backward 改为转发入口。
- 用户新增“原文件不修改”的约束后，取消兼容入口替换，保留原实现作为独立学习参考。
