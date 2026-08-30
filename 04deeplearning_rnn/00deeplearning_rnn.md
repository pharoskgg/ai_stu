# 基础 RNN 的正向传播与反向传播

## 1. 符号约定

假设序列长度为 $T$，输入维度为 $D$，隐藏状态维度为 $H$。本文采用列向量的数学约定：

$$
x_t\in\mathbb{R}^{D},\qquad
h_t\in\mathbb{R}^{H}
$$

参数为：

$$
W_x\in\mathbb{R}^{H\times D},\qquad
W_h\in\mathbb{R}^{H\times H},\qquad
b\in\mathbb{R}^{H}
$$

其中：

- $W_x$ 是输入到隐藏状态的权重；
- $W_h$ 是前一个隐藏状态到当前隐藏状态的循环权重；
- $b$ 是偏置；
- $h_0$ 是初始隐藏状态，通常初始化为零。

如果程序使用行向量存储输入和隐藏状态，矩阵乘法中的转置位置会与本文相反，但计算含义完全相同。

为了避免混淆，本文使用：

- $\ell_t$ 表示第 $t$ 个时间步产生的局部损失；
- $J$ 表示整条序列的总损失。

若每个时间步都产生损失，则：

$$
J=\sum_{t=1}^{T}\ell_t
$$

若只有最后一个隐藏状态产生损失，则可以认为：

$$
\ell_1=\ell_2=\cdots=\ell_{T-1}=0,\qquad J=\ell_T
$$

## 2. 正向传播

RNN 在第 $t$ 个时间步首先计算线性部分：

$$
z_t=W_xx_t+W_hh_{t-1}+b
$$

然后通过双曲正切激活函数得到隐藏状态：

$$
h_t=\tanh(z_t)
$$

因此完整的时间展开为：

$$
h_1=\tanh(W_xx_1+W_hh_0+b)
$$

$$
h_2=\tanh(W_xx_2+W_hh_1+b)
$$

$$
\cdots
$$

$$
h_T=\tanh(W_xx_T+W_hh_{T-1}+b)
$$

每个隐藏状态不仅可以作为当前时间步的输出，还会参与下一个隐藏状态的计算。因此，$h_t$ 通常有两条下游路径：

1. 影响当前时间步的损失 $\ell_t$；
2. 通过 $h_{t+1}$ 影响所有未来损失。

这两条路径正是反向传播时隐藏状态梯度需要相加的原因。

## 3. 隐藏状态梯度为什么需要相加

令：

$$
g_t=\frac{\partial J}{\partial h_t}
$$

它表示总损失对第 $t$ 个隐藏状态的梯度。

再令：

$$
r_t=\frac{\partial \ell_t}{\partial h_t}
$$

它表示当前时间步的损失直接传给 $h_t$ 的梯度。若当前时间步没有产生损失，则 $r_t=0$。

因为 $h_t$ 同时影响当前损失和未来损失，所以总梯度为：

$$
\boxed{
g_t
=
\frac{\partial J}{\partial h_t}
=
\underbrace{\frac{\partial \ell_t}{\partial h_t}}_{\text{当前时间步直接传来的梯度}}
+
\underbrace{
\left(\frac{\partial h_{t+1}}{\partial h_t}\right)^T
\frac{\partial J}{\partial h_{t+1}}
}_{\text{所有未来时间步传回来的梯度}}
}
$$

也可以简写为：

$$
\boxed{g_t=r_t+g_t^{\mathrm{future}}}
$$

这里的加法并不是来自正向公式

$$
z_t=W_xx_t+W_hh_{t-1}+b
$$

中的加法，而是因为同一个变量 $h_t$ 沿着两条不同路径影响总损失。一个变量通过多条路径影响损失时，各条路径传回来的梯度必须相加。

## 4. 用三个时间步展开梯度

假设：

$$
J=\ell_1+\ell_2+\ell_3
$$

时间关系为：

$$
h_1\longrightarrow h_2\longrightarrow h_3
$$

并且 $h_1,h_2,h_3$ 分别影响 $\ell_1,\ell_2,\ell_3$。

对于 $h_2$，它会影响当前损失 $\ell_2$，还会通过 $h_3$ 影响未来损失 $\ell_3$：

$$
\frac{\partial J}{\partial h_2}
=
\frac{\partial \ell_2}{\partial h_2}
+
\left(\frac{\partial h_3}{\partial h_2}\right)^T
\frac{\partial \ell_3}{\partial h_3}
$$

对于 $h_1$，直接展开可得：

$$
\frac{\partial J}{\partial h_1}
=
\frac{\partial \ell_1}{\partial h_1}
+
\left(\frac{\partial h_2}{\partial h_1}\right)^T
\frac{\partial \ell_2}{\partial h_2}
+
\left(\frac{\partial h_2}{\partial h_1}\right)^T
\left(\frac{\partial h_3}{\partial h_2}\right)^T
\frac{\partial \ell_3}{\partial h_3}
$$

后面两项正好可以合并为 $h_2$ 接收到的总梯度，因此：

$$
\frac{\partial J}{\partial h_1}
=
\frac{\partial \ell_1}{\partial h_1}
+
\left(\frac{\partial h_2}{\partial h_1}\right)^T
\frac{\partial J}{\partial h_2}
$$

这说明 $\frac{\partial J}{\partial h_{t+1}}$ 已经汇总了第 $t+1$ 个时间步及其后面所有时间步的影响，所以不需要再次把所有未来损失逐项展开。

## 5. 为什么有的损失带下标，有的不带下标

$\ell_t$ 是局部损失，只属于第 $t$ 个时间步；$J$ 是总损失，包含整条序列的所有局部损失：

$$
J=\ell_1+\ell_2+\cdots+\ell_T
$$

因此：

$$
\frac{\partial \ell_t}{\partial h_t}
$$

只表示当前时间步直接传给 $h_t$ 的梯度，而：

$$
\frac{\partial J}{\partial h_{t+1}}
$$

表示总损失通过 $h_{t+1}$ 汇总后传回来的梯度。由于过去的损失 $\ell_1,\ldots,\ell_t$ 不依赖未来状态 $h_{t+1}$，它实际上只包含未来损失的贡献。

如果只有最后一个隐藏状态产生损失，那么对于 $t<T$：

$$
r_t=\frac{\partial \ell_t}{\partial h_t}=0
$$

此时较早隐藏状态的梯度完全来自未来：

$$
g_t=
\left(\frac{\partial h_{t+1}}{\partial h_t}\right)^Tg_{t+1}
$$

通用推导仍然保留加法，是因为它还需要覆盖每个时间步都有输出和损失的情况。

## 6. 中间的 $h_{t+1}$ 能不能约掉

链式法则中的表达式：

$$
\left(\frac{\partial h_{t+1}}{\partial h_t}\right)^T
\frac{\partial J}{\partial h_{t+1}}
$$

看起来像普通分数相乘，但导数不是普通分数。对于向量，$\frac{\partial h_{t+1}}{\partial h_t}$ 是 Jacobian 矩阵，上面的乘法表示线性映射的复合，而不是分数的乘除。

在一维且只有一条计算路径时，把中间变量“约掉”可以作为记忆链式法则的方式。但在这里，这一项只表示经过 $h_{t+1}$ 这条未来路径传回来的梯度，并不是 $h_t$ 接收到的全部梯度。

例如，令：

$$
y=x^2,\qquad J=y+3x
$$

经过 $y$ 这条路径传回来的梯度是：

$$
\frac{\partial J}{\partial y}
\frac{\partial y}{\partial x}
=2x
$$

但总梯度为：

$$
\frac{\partial J}{\partial x}=2x+3
$$

其中 $3$ 来自 $x$ 直接影响 $J$ 的另一条路径。如果直接把中间变量约掉并把结果理解为总梯度，就会漏掉这条路径。

RNN 中同理：经过 $h_{t+1}$ 的梯度只是未来路径的贡献，还必须加上当前损失直接传来的梯度。

## 7. tanh 的反向传播

当前隐藏状态为：

$$
h_t=\tanh(z_t)
$$

tanh 的导数为：

$$
\frac{\partial h_t}{\partial z_t}
=1-\tanh^2(z_t)
=1-h_t^2
$$

定义线性部分的梯度：

$$
\delta_t=\frac{\partial J}{\partial z_t}
$$

则：

$$
\boxed{
\delta_t=g_t\odot(1-h_t^2)
}
$$

其中 $\odot$ 表示逐元素相乘，不是矩阵乘法。

## 8. 未来隐藏状态梯度的具体形式

下一个时间步的线性部分为：

$$
z_{t+1}=W_xx_{t+1}+W_hh_t+b
$$

损失对 $h_t$ 的未来路径梯度为：

$$
g_t^{\mathrm{future}}
=
W_h^T\delta_{t+1}
$$

因此隐藏状态的完整递推公式是：

$$
\boxed{
g_t=r_t+W_h^T\delta_{t+1}
}
$$

再经过当前时间步的 tanh：

$$
\boxed{
\delta_t=
\left(r_t+W_h^T\delta_{t+1}\right)
\odot(1-h_t^2)
}
$$

反向传播从 $t=T$ 开始，依次计算 $T-1,T-2,\ldots,1$。在最后一个时间步之后不存在新的隐藏状态，因此边界条件可以写为：

$$
\delta_{T+1}=0
$$

## 9. 参数和输入的梯度

由：

$$
z_t=W_xx_t+W_hh_{t-1}+b
$$

可以得到第 $t$ 个时间步对输入权重的梯度贡献：

$$
\left.\frac{\partial J}{\partial W_x}\right|_t
=\delta_tx_t^T
$$

对循环权重的梯度贡献：

$$
\left.\frac{\partial J}{\partial W_h}\right|_t
=\delta_th_{t-1}^T
$$

对偏置的梯度贡献：

$$
\left.\frac{\partial J}{\partial b}\right|_t
=\delta_t
$$

由于同一组参数在所有时间步共享，总参数梯度必须把所有时间步的贡献相加：

$$
\boxed{
\frac{\partial J}{\partial W_x}
=\sum_{t=1}^{T}\delta_tx_t^T
}
$$

$$
\boxed{
\frac{\partial J}{\partial W_h}
=\sum_{t=1}^{T}\delta_th_{t-1}^T
}
$$

$$
\boxed{
\frac{\partial J}{\partial b}
=\sum_{t=1}^{T}\delta_t
}
$$

损失对输入的梯度为：

$$
\boxed{
\frac{\partial J}{\partial x_t}=W_x^T\delta_t
}
$$

如果 $h_0$ 是固定的零向量，它不是可训练参数，所以通常不需要使用 $\frac{\partial J}{\partial h_0}$ 更新任何内容；但这个梯度仍可以按同样方式计算出来。

## 10. 完整的 BPTT 推导关系

基础 RNN 在每个时间步的反向传播可以概括为以下四组关系。

隐藏状态接收到的总梯度：

$$
g_t=r_t+W_h^T\delta_{t+1}
$$

通过激活函数后的梯度：

$$
\delta_t=g_t\odot(1-h_t^2)
$$

参数梯度：

$$
\frac{\partial J}{\partial W_x}
=\sum_{t=1}^{T}\delta_tx_t^T,\qquad
\frac{\partial J}{\partial W_h}
=\sum_{t=1}^{T}\delta_th_{t-1}^T,\qquad
\frac{\partial J}{\partial b}
=\sum_{t=1}^{T}\delta_t
$$

输入梯度：

$$
\frac{\partial J}{\partial x_t}=W_x^T\delta_t
$$

这就是随时间反向传播（Backpropagation Through Time，BPTT）的核心。

## 11. 梯度消失和梯度爆炸

假设损失只产生于最后一个时间步。忽略当前时间步的直接损失后，隐藏状态梯度需要不断经过循环连接：

$$
g_t=
\left(\frac{\partial h_{t+1}}{\partial h_t}\right)^T
g_{t+1}
$$

对于 tanh RNN，定义：

$$
D_k=\operatorname{diag}(1-h_k^2)
$$

则：

$$
\frac{\partial h_k}{\partial h_{k-1}}=D_kW_h
$$

从最后一个隐藏状态连续传播到较早的隐藏状态，会得到一连串 Jacobian 矩阵的乘积：

$$
g_t=
\left(D_{t+1}W_h\right)^T
\left(D_{t+2}W_h\right)^T
\cdots
\left(D_TW_h\right)^Tg_T
$$

如果这些 Jacobian 的范数长期小于 $1$，连乘结果会趋近于零，产生梯度消失；如果其范数长期大于 $1$，连乘结果可能迅速增大，产生梯度爆炸。

由于：

$$
0\leq 1-h_k^2\leq 1
$$

而且 tanh 饱和时 $|h_k|$ 接近 $1$，此时 $1-h_k^2$ 接近 $0$，所以 tanh RNN 很容易出现梯度消失。

## 12. 梯度消失或爆炸是否只影响隐藏状态梯度

最先沿时间反复连乘的是隐藏状态梯度 $g_t$，因此它是问题的传播核心。但参数梯度、偏置梯度和输入梯度都依赖 $\delta_t$：

$$
\delta_t=g_t\odot(1-h_t^2)
$$

而参数梯度又由 $\delta_t$ 计算：

$$
\frac{\partial J}{\partial W_h}
=\sum_t\delta_th_{t-1}^T,\qquad
\frac{\partial J}{\partial W_x}
=\sum_t\delta_tx_t^T
$$

所以因果关系是：

$$
\text{循环 Jacobian 反复连乘}
\longrightarrow
g_t\text{ 消失或爆炸}
\longrightarrow
\delta_t\text{ 消失或爆炸}
\longrightarrow
\text{参数梯度受到影响}
$$

需要注意，参数在所有时间步共享，参数总梯度是各时间步贡献的和。即使很早时间步的梯度贡献已经消失，靠近最终损失的几个时间步仍可能产生正常梯度。因此，参数总梯度不一定整体为零，但模型无法把很久以前的信息与当前损失联系起来，也就难以学习长期依赖。

梯度爆炸则可能使隐藏状态梯度和参数梯度都变得非常大，造成参数更新不稳定甚至出现数值溢出。梯度裁剪可以限制梯度范数，缓解梯度爆炸，但不能从根本上解决梯度消失。

## 13. 核心理解

基础 RNN 反向传播最重要的逻辑可以概括为：

1. $h_t$ 同时影响当前损失和未来隐藏状态，所以它接收到的梯度是两条路径梯度之和；
2. 未来梯度通过循环权重和激活函数导数逐步向前传播；
3. 每个时间步共享同一组参数，所以参数梯度要对所有时间步的贡献求和；
4. 隐藏状态梯度沿时间反复乘以循环 Jacobian，构成梯度消失和梯度爆炸的根源；
5. 隐藏状态梯度的问题会继续传给 $\delta_t$，最终影响各项参数梯度以及模型学习长期依赖的能力。
