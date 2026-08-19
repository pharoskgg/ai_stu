# 深度学习进阶 (Deep Learning Improvement)

## 1. 机器学习策略 (ML Strategy)

### 1.1 训练集/验证集/测试集 (Train/Dev/Test sets)

在机器学习中，通常将样本分成**训练集**、**验证集**和**测试集**三部分：

- **训练集 (Training Set)**: 用于训练模型参数（权重w和偏置b）
- **验证集 (Development/Validation Set)**: 用于调整超参数、选择模型、防止过拟合
- **测试集 (Test Set)**: 用于最终评估模型的泛化能力

#### 数据划分比例

- **小数据集** (< 10万条): 传统划分比例
  - 训练集: 60%
  - 验证集: 20%
  - 测试集: 20%

- **大数据集** (> 100万条): 验证集和测试集占比可降低
  - 训练集: 98%
  - 验证集: 1%
  - 测试集: 1%

#### 重要原则

1. **验证集和测试集应来自同一分布**: 确保评估的有效性
2. **测试集只使用一次**: 避免根据测试结果反复调整模型导致过拟合
3. **数据量充足时**: 验证集和测试集可以占到数据总量的20%或10%以下

---

### 1.2 偏差/方差 (Bias/Variance)

诊断模型的两种主要问题：

#### 高偏差 (High Bias) - 欠拟合 (Underfitting)

- **表现**: 
  - 训练集准确率低
  - 验证集准确率也低
  - 两者差距不大
  
- **原因**: 模型太简单，无法捕捉数据的复杂模式

- **解决方案**:
  - 增加网络层数或神经元数量
  - 增加特征数量
  - 减少正则化强度
  - 尝试更复杂的模型架构

#### 高方差 (High Variance) - 过拟合 (Overfitting)

- **表现**:
  - 训练集识别率高（接近100%）
  - 验证集识别率低
  - 两者差距很大
  
- **原因**: 模型过于复杂，记住了训练数据的噪声而非规律

- **解决方案**:
  - 增加训练数据量
  - 使用正则化技术 (L1/L2 regularization)
  - Dropout随机失活
  - 简化模型结构
  - 早停法 (Early Stopping)

#### 理想状态

- 训练集准确率高
- 验证集准确率也高
- 两者差距小

---


### 1.3 基本机器学习设置 (Basic ML Setup)

#### 误差分析流程

1. **建立基线 (Baseline)**: 确定人类水平或当前最佳性能
2. **分析错误**: 手动检查验证集上的错误案例
3. **优先解决**: 找出最容易改进且影响最大的问题
4. **快速迭代**: 快速实验，根据结果调整策略


- 如果碰到高偏差（欠拟合），通常是增加神经网络的隐藏层层数、神经元个数，训练时间延长，选择其它更复杂的NN模型等
- 其次，减少高方差（过拟合）的方法通常是增加训练样本数据，但有时无法获得更多的数据，进行正则化（Regularization）来减少过拟合

#### 正交化 (Orthogonalization)

- 每个步骤只解决一个具体问题
- 避免同时调整多个因素
- 便于定位问题和理解因果关系

---

### 1.4 正则化 Regularization
模型为了把训练集拟合得很准，可能会把某些权重调得特别大；正则化会对“大权重”收取额外代价，迫使模型采用更温和、更简单的规则。

在成本函数中添加权重的平方和：

$$J(w,b) = \frac{1}{m}\sum L(\hat{y}, y) + \frac{\lambda}{2m}\sum ||w||^2$$

#### lambda
- ${\lambda}$ 决定“往 0 拉”的力度：
- \(\lambda=0\)：不约束，模型可能过拟合；
- \(\lambda\) 适中：过滤噪声，泛化更好；
- \(\lambda\) 太大：权重几乎都接近 0，模型表达能力不足，变成欠拟合

#### 梯度更新

$$w := w - \alpha\left[\frac{\partial J}{\partial w} + \frac{\lambda}{m}w\right]$$

等价于：

$$w := (1 - \frac{\alpha\lambda}{m})w - \alpha\frac{\partial J}{\partial w}$$

所以也叫**权重衰减**。

#### 效果

- 抑制大权重值
- 使模型更简单平滑
- λ越大，正则化越强


#### 原因
正则化λ设置得足够大，权重矩阵W被设置为接近于0的值，就是把多隐层单元的权重设为0，于是基本上消除了那些隐藏单元的许多影响。这种情况下，复杂神经网络就被简化成很小的网络，同时深度很大，它会使这个网络从过拟合的状态更接近于左图的高偏差状态。但是λ有个中间值，使得接近“just right”的中间状态。因此，选择合适大小的λ值，就能够同时避免高偏差（high bias）和高方差（high variance），得到最佳模型。

#### 例子
训练时不再只追求“预测误差低”，而是同时追求：
预测得准，并且不要依赖特别大的权重。

例如两个模型训练误差都很低：
- 模型 A：权重为 \([0.3,-0.5,0.2]\)，规律平滑；
- 模型 B：权重为 \([20,-15,8]\)，对输入的微小变化很敏感。

不加正则化时两者可能都能被选中；加 L2 正则化后，B 的 \(W^2\) 非常大，会被额外处罚，因此更倾向 A。这样模型较不容易“死记”训练数据中的偶然噪声，验证集表现通常更好。

### 1.5 Dropout 随机失活

当网络过拟合时，dropout 按概率随机"杀死"部分神经元，使网络不依赖某些特定神经元，从而降低过拟合。

#### 代码实现（Inverted Dropout）

```python
# 第1步：生成随机掩码，keep_prob 为保留概率
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob

# 第2步：置零被杀死的神经元
a3 = np.multiply(a3, d3)  # 等价于 a3 *= d3

# 第3步：除以 keep_prob，修正期望值
a3 /= keep_prob
```

#### 为什么要除以 keep_prob？

dropout 置零了一部分神经元，**输出期望缩小了**。除以 `keep_prob` 是为了把期望修正回原来的水平。

| 场景 | 期望输出 |
|------|---------|
| 不做 dropout | E[a] = Σaᵢ |
| dropout 不除回来 | E[a] = keep_prob × Σaᵢ（缩小了） |
| dropout 除回来 | E[a] = keep_prob × Σaᵢ / keep_prob = Σaᵢ（不变） |

举例：4个神经元输出 [2, 4, 6, 8]，keep_prob = 0.5

- 不做 dropout：总和 = 20
- dropout 不除回来：平均杀掉一半，期望总和 = 20 × 0.5 = 10
- dropout 除回来：期望总和 = 10 / 0.5 = 20，和原来一致

#### 为什么测试时不做 dropout？

测试时所有神经元都保留，输出本身就是 Σaᵢ。训练时已经除回来了，数值尺度天然一致，测试代码无需任何额外处理。

#### 梯度计算需要改变吗？

不需要。`mask / keep_prob` 是前向计算图的一部分，反向传播时链式法则自动处理。

**前向：**

```
a_drop = a ⊙ mask / keep_prob
```

**反向链式推导：**

```
∂L     ∂L        ∂a_drop
─── = ────── · ────────
∂a     ∂a_drop      ∂a

∂L        ∂a_drop
───── = dz_drop · ────────
∂a               ∂a

∂a_drop/∂a = mask / keep_prob    （常数因子，不随 a 变化）

∴ da = dz_drop · mask / keep_prob
```

`mask / keep_prob` 在前向时已经确定，反向时直接乘上去即可，**无需手动修改任何梯度公式**。

#### dW 会包含 keep_prob 吗？

会，但不需要手动加。假设第3层做了 dropout，第4层是输出层：

**前向：**

```
a3_drop = a3 ⊙ mask / keep_prob
z4 = W4^T · a3_drop + b4
```

**dW4 推导：**

```
dW4 = ∂L/∂W4
    = ∂L/∂z4 · ∂z4/∂W4
    = dz4 · a3_drop^T
    = dz4 · (a3 ⊙ mask / keep_prob)^T
    = dz4 · (a3 ⊙ mask)^T / keep_prob
```

keep_prob 通过 a3_drop 间接进入了 dW4，链式法则自动带入，无需手动处理。

**各参数与 keep_prob 的关系：**

| 参数 | 推导式 | keep_prob 来源 |
|------|--------|---------------|
| da3 | dz4 · W4 ⊙ mask / keep_prob | 直接包含 |
| dW4 | dz4 · a3_drop^T | 通过 a3_drop 间接包含 |
| db4 | Σ(dz4) | 不包含（与 a3_drop 无关） |

#### 理解dropout
dropout 是一种 prevent overfitting 的方法，通过随机“杀死”部分神经元，使网络不依赖某些特定神经元，从而降低过拟合。

### 其他正则化
1. 对于图片
- 随机裁剪：随机裁剪图片，使图片尺寸变小，从而增加数据量，并避免过拟合。
- 随机旋转：随机旋转图片，使图片角度变化，从而增加数据量，并避免过拟合。
- 造假数据：造假数据，使数据量变大，从而增加数据量，并避免过拟合。

2. early stopping：在训练过程中，如果验证集的准确率开始下降，就停止训练，从而避免过拟合。


### 标准化输入 (Input Normalization)
对输入数据进行标准化处理，使得不同特征的数值范围相近，从而加速梯度下降的收敛速度。这个过程也叫**特征缩放**（Feature Scaling），包含两个步骤：

1. **零均值化**（Zero-mean）：将每个特征的均值调整为0。
   
   对每个特征 $j$，计算均值：$\mu_j = \frac{1}{m}\sum_{i=1}^{m} x_j^{(i)}$
   
   零均值化：$x_j := x_j - \mu_j$

2. **标准化**（Standardization）：将每个特征的方差调整为1。
   
   对每个特征 $j$，计算标准差：$\sigma_j = \sqrt{\frac{1}{m}\sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2}$
   
   标准化：$x_j := \frac{x_j}{\sigma_j}$

举例：两个特征

特征	数据	均值	标准差 σ
特征A	[9, 10, 11]	10	≈ 0.8（很集中）
特征B	[0, 10, 20]	10	≈ 8.2（很分散）


除 σ 本质上就是「按当前数据的松散程度等比缩放」：

数据很散（σ 大，比如 8.2）→ 除以一个大数 → 被压缩回标准尺度
数据很集中（σ 小，比如 0.8）→ 除以一个小数 → 被拉开放大

综合起来，标准化公式为：

$$x_j^{\text{norm}} = \frac{x_j - \mu_j}{\sigma_j}$$


![alt text](asset/normaliz_set.png)

#### why 需要标准化？

1. **加速梯度下降收敛**：当不同特征的数值范围差异很大时（如一个特征范围是 0-1，另一个是 0-1000），损失函数的等高线会呈现狭长的椭圆形，梯度下降会在峡谷中来回震荡，收敛很慢。标准化后，等高线接近圆形，梯度下降可以直接朝向最优解，收敛速度大幅提升。

2. **避免数值问题**：某些特征的值过大可能导致梯度爆炸或消失，标准化可以将所有特征缩放到相似的数值范围（通常是均值为0，方差为1），使训练更加稳定。

3. **提高模型性能**：对于基于距离的算法（如 KNN、SVM）和使用梯度下降的神经网络，标准化可以确保所有特征对模型的贡献相对均衡，避免某些特征因为数值范围大而主导模型。

#### 注意事项

- **训练集和测试集使用相同的 $\mu$ 和 $\sigma$**：只能用训练集计算均值和标准差，然后应用到验证集和测试集，避免数据泄露。
- **不是所有情况都需要**：如果所有特征已经在相似的范围内（如像素值 0-255），标准化的效果可能不明显。


### 权重初始化
z = w1x1 + w2x2 + .... + wnxn

n越大，希望w_i越小，避免梯度消失或者爆炸，一种合理的方法是设置w_i = 1/ n n为神经元的输入特征数量；它不能比1大很多，也不能比1小很多，还有一些其他的初始化方式也是相同的目的

w[l] = np.random.randn(n[l],n[l-1])*np.sqrt(1/n[l-1])

### minibatch
批量梯度下降的缺点是每次迭代都需要计算所有样本的梯度，这非常耗时。因此，我们可以将数据集划分为多个小批量，每次迭代只计算小批量的梯度，从而提高效率。

要决定的变量之一是mini-batch的大小，m就是训练集的大小，极端情况下：如果为m，即为batch梯度下降法（BGD），只包含一个子集；如果为1，即为随机梯度下降法（SGD），每次只处理一个训练样本

数量m不太大时，例如m<2000，建议直接使用Batch gradient descent


###指数加权平均(EMA)

### 三类平均值公式与对比

#### 1. 计算公式
**算术平均值（MA）**
$$
MA = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

**普通加权平均值（WA）**
$$
WA = \frac{\displaystyle\sum_{i=1}^{n} w_i x_i}{\displaystyle\sum_{i=1}^{n} w_i}
$$

**指数加权平均值（EMA）**
递推公式：
$$
EMA_t = \alpha \, x_t + (1-\alpha) \, EMA_{t-1}
$$
权重展开形式：
$$
EMA_t = \alpha x_t + \alpha(1-\alpha)x_{t-1} + \alpha(1-\alpha)^2 x_{t-2} + \dots
$$
系数换算：
$$
\alpha = \frac{2}{n+1}
$$

当 alpha 越大时，权重越小，越不敏感；当 alpha 越小时，权重越小，越敏感，比如alpha=0.5，则权重为0.5，0.25，0.125，0.0625，0.03125，0.015625 衰减的非常快也就越震荡；越大则越平坦

#### 2. 特性对比表
| 对比维度 | 算术平均 (MA) | 普通加权平均 (WA) | 指数加权平均 (EMA) |
| ---- | ---- | ---- | ---- |
| 权重规则 | 所有权重相等 | 人为设置静态固定权重 | 权重随时间指数衰减，近大远小 |
| 计算形式 | 固定窗口、静态求和 | 固定样本、静态求和 | 递推计算，无硬性截断窗口 |
| 新旧数据影响 | 新旧数据影响力一致 | 权重固定，影响力不变 | 最新数据影响最大，旧数据逐步弱化 |
| 响应速度 | 慢，滞后明显 | 取决于人工权重 | 快，紧跟当前数据变化 |
| 平滑效果 | 中等平滑 | 由人工权重决定 | 平滑且保留趋势，毛刺少 |
| 数据取舍 | 只取最近 n 个数据 | 选定样本集 | 理论包含全部历史，远期权重趋近于 0 |


### 指数加权平均的偏差修正
指数加权平均在初期（前几步）计算结果严重偏低、不准

#### 1. 原始 EMA 递推公式
$$
v_t = \beta v_{t-1} + (1-\beta)\theta_t
$$

- $v_t$：第 $t$ 步指数加权平均结果
- $\beta$：衰减系数（$0<\beta<1$）
- $v_{t-1}$：上一步加权平均结果
- $\theta_t$：第 $t$ 步原始观测值（深度学习中为梯度）
- 初始化：$v_0 = 0$

#### 2. 初始偏差问题
由于默认 $v_0=0$，迭代前期计算结果会被低估，存在明显偏差。

以 $\beta=0.9$ 为例：
$$
\begin{align*}
v_1 &= 0.9 \cdot 0 + 0.1\theta_1 = 0.1\theta_1 \\
v_2 &= 0.9 v_1 + 0.1\theta_2
\end{align*}
$$

前期数值远小于真实均值，会导致优化器梯度估计不准、参数更新震荡。

#### 3. 偏差修正公式
$$
\hat{v}_t = \frac{v_t}{1 - \beta^t}
$$

- $\hat{v}_t$：修正后的无偏估计值
- $t$：当前迭代步数

#### 修正原理
1. 迭代前期（$t$ 很小）：$1-\beta^t \ll 1$，除以小数放大数值，抵消初始低估；
2. 迭代后期（$t \to +\infty$）：$\beta^t \to 0$，$1-\beta^t \to 1$，修正自动失效，不影响正常计算。

#### 4. 数值演示（$\beta=0.9$）
$$
\begin{align*}
\hat{v}_1 &= \frac{0.1\theta_1}{1-0.9^1} = \theta_1 \\
1-0.9^{10} &\approx 0.651 \\
1-0.9^{50} &\approx 0.995
\end{align*}
$$

- 第1步：完全修正，还原真实值
- 第10步：仍存在小幅修正
- 第50步：修正效果基本消失

### RMSProp (root mean square propagation)
加速梯度下降，通过计算梯度的平方根来控制学习率，从而避免梯度爆炸RMSProp 的核心是计算梯度的平方根，而不是梯度的绝对值。**使得学习率的调整更加平滑，能自适应调整每个参数大学习率**。公式如下:

Sdw = $\beta Sdw + (1-\beta) dW^2$

Sdb = $\beta Sdb + (1-\beta) db^2$

W = W - $\alpha \frac{dW}{\sqrt{Sdw + \epsilon}}$

b = b - $\alpha \frac{db}{\sqrt{Sdb + \epsilon}}$

dW^2的作用是让梯度下降时，当梯度变大时，学习率变小，当梯度变小时，学习率变大。例如：
dw梯度等于0.1，则Sdw等于0.1^2=0.01，分母变小相当于学习率变大，同理dw梯度等于2时，Sdw等于2^2=4，分母变大相当于学习率变小。
- 在加速时踩刹车，在减速时踩油门


### Adam
Adam 的核心是计算梯度的均值和方差，从而控制学习率。公式如下:
Adam = RMSProp + Momentum
1. 初始化 
 - Vdw = 0, Sdw = 0, Vdb = 0, Sdb = 0
2. 计算Momemtum指数加权平均数
- Vdw = $\beta_1 Vdw + (1-\beta_1) dW$
- Vdb = $\beta_1 Vdb + (1-\beta_1) db$
3. 使用RMSProp进行更新
- Sdw = $\beta_2 Sdw + (1-\beta_2) dW^2$
- Sdb = $\beta_2 Sdb + (1-\beta_2) db^2$
4. 使用Adam算法，一般要计算偏差修正
$$
V_{dw}^{corrected} = \frac{V_{dw}}{1 - \beta_1^t}, \quad
V_{db}^{corrected} = \frac{V_{db}}{1 - \beta_1^t}
$$

$$
S_{dw}^{corrected} = \frac{S_{dw}}{1 - \beta_2^t}, \quad
S_{db}^{corrected} = \frac{S_{db}}{1 - \beta_2^t}
$$

5. 更新参数
- W = W - $\alpha \frac{V_{dw}^{corrected}}{\sqrt{S_{dw}^{corrected} + \epsilon}}$
- b = b - $\alpha \frac{V_{db}^{corrected}}{\sqrt{S_{db}^{corrected} + \epsilon}}$


一般使用默认值（default value），β_1=0.9，β_2=0.999（Adam论文的作者，也就是Adam算法的发明者推荐），ε=10^-8（Adam论文的作者推荐）。然后尝试不同的学习率，看看哪个效果最好。但也可以调整β_1和β_2


### 学习率衰减
1. 线性衰减
2. 指数衰减

例如：
$\alpha = 0.95^{epoch}$

### 鞍点
设想在2万维的空间中，要得到局部最优，所有2万个方向都是凸函数，这发生的几率也许很小（2^-20000）。因此在高维空间，更可能碰到鞍点，而不是局部最优点。即某个方向上可能已经最优里，但是沿着这个方向，其他方向的函数值可能比这个方向的函数值要小。损失函数定义在更高维度

### 超参数调参选择
1. 学习因子α是最重要的超参数，也是需要重点调试的超参数。
2. 动量梯度下降因子beta、各隐藏层神经元个数hidden units和mini-batch size的重要性仅次于alpha。
3. 然后就是神经网络层数layers和学习因子下降参数learning rate decay。
4. 最后，Adam算法的三个参数bata1,bata2,sigma一般常设置为0.9，0.999和10^-8，不需要反复调试。

当然，这里超参数重要性的排名并不是绝对的，具体情况，具体分析；

### 超参数范围选择
1. 网格法:  $\alpha$ 和 Adam中的beta 弄成 5 * 5的表格挨个试，刚开始范围可以大一点，然后再对效果好的范围进行选择。即:由粗糙到精细的策略
2. 随机法: 选择合适的标尺 log对数搜索标尺，比如学习率: 分别依次取0.0001，0.001，0.01，0.1，1，在对数轴上均匀随机取点，这样，在0.0001到0.001之间，就会有更多的搜索资源可用，还有在0.001到0.01之间等等

### 调参实践
1. 一种是照看一个模型（babysit one model），通常有庞大的数据组，但是没有许多计算资源或足够的CPU和GPU，这样的话一次试验一个模型或者一小批模型，然后每天花时间观察它，不断调整参数
2. 一种是照看一个模型（babysit one model），通常有庞大的数据组，但是没有许多计算资源或足够的CPU和GPU，这样的话一次试验一个模型或者一小批模型，然后每天花时间观察它，不断调整参数

总结，这两种方式的选择，是由你拥有的计算资源决定的

### 归一化网络的激活函数 Normalizing activations in a network
$\mu = \frac{1}{M}\sum_{i=1}^M z_i$

$\sigma^2 = \frac{1}{M}\sum_{i=1}^M (a_i - \mu)^2$

$z_{\text{norm}}^{(i)} = \frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$

$\tilde{z}^{(i)} = \gamma z_{\text{norm}}^{(i)} + \beta$

其中，γ和β是模型的学习参数（learnable parameters），可以使用梯度下降算法或者Momentum、Nesterov、Adam等更新它们

Batch归一化的作用是它适用的归一化过程，不只是输入层（input layer），甚至同样适用于神经网络中的深度隐藏层（hidden layer）

### Batch Norm为什么奏效
- 联想输入特征归一化，输入特征值归一化后获得类似范围的值，可以加速学习
- 它可以使权重比你的网络更滞后或更深层，比如，第10层的权重更能经受得住变化
- 限制了权重范围，可以防止权重过小或者过大，从而防止梯度消失或者爆炸
- 使这些值变得更稳定（stable），神经网络的之后层就会有更坚实的基础。即使使输入分布改变了一些，它会改变得更少。它做的是当前层保持学习，当改变时，迫使后层适应的程度减小了，你可以这样想，它减弱了前层参数的作用与后层参数的作用之间的联系，它使得网络每层都可以自己学习，稍稍独立于其它层，这有助于加速整个网络的学习。
- 轻微正则化，如果使用minibatch，对各个隐层添加了随机噪声，batch越小，正则化效果越明显

### Batch Norm 反向传播 (Backpropagation)

在反向传播中，我们需要计算损失函数 $L$ 对 Batch Norm 各参数的梯度。假设我们已经从后一层得到了 $\frac{\partial L}{\partial \tilde{z}^{(i)}}$（即 $d\tilde{z}$）。


#### 反向传播推导

**步骤1：计算 $\frac{\partial L}{\partial \gamma}$ 和 $\frac{\partial L}{\partial \beta}$**

$$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^m \frac{\partial L}{\partial \tilde{z}^{(i)}} \cdot z_{\text{norm}}^{(i)}$$

$$\frac{\partial L}{\partial \beta} = \sum_{i=1}^m \frac{\partial L}{\partial \tilde{z}^{(i)}}$$

**步骤2：计算 $\frac{\partial L}{\partial z_{\text{norm}}^{(i)}}$**

$$\frac{\partial L}{\partial z_{\text{norm}}^{(i)}} = \frac{\partial L}{\partial \tilde{z}^{(i)}} \cdot \gamma$$

**步骤3：计算 $\frac{\partial L}{\partial \sigma^2}$**

由于 $z_{\text{norm}}^{(i)}$ 依赖于 $\sigma^2$，根据链式法则：

$$\frac{\partial L}{\partial \sigma^2} = \sum_{i=1}^m \frac{\partial L}{\partial z_{\text{norm}}^{(i)}} \cdot \frac{\partial z_{\text{norm}}^{(i)}}{\partial \sigma^2}$$

其中：

$$\frac{\partial z_{\text{norm}}^{(i)}}{\partial \sigma^2} = (z^{(i)} - \mu) \cdot \left(-\frac{1}{2}\right)(\sigma^2 + \epsilon)^{-3/2}$$

因此：

$$\frac{\partial L}{\partial \sigma^2} = \sum_{i=1}^m \frac{\partial L}{\partial z_{\text{norm}}^{(i)}} \cdot (z^{(i)} - \mu) \cdot \left(-\frac{1}{2}\right)(\sigma^2 + \epsilon)^{-3/2}$$

**步骤4：计算 $\frac{\partial L}{\partial \mu}$**

$\mu$ 通过两条路径影响损失：
1. 直接通过 $z_{\text{norm}}^{(i)}$
2. 通过 $\sigma^2$

$$\frac{\partial L}{\partial \mu} = \sum_{i=1}^m \frac{\partial L}{\partial z_{\text{norm}}^{(i)}} \cdot \frac{\partial z_{\text{norm}}^{(i)}}{\partial \mu} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial \mu}$$

其中：

$$\frac{\partial z_{\text{norm}}^{(i)}}{\partial \mu} = -\frac{1}{\sqrt{\sigma^2 + \epsilon}}$$

$$\frac{\partial \sigma^2}{\partial \mu} = \frac{1}{m}\sum_{i=1}^m 2(z^{(i)} - \mu) \cdot (-1) = -\frac{2}{m}\sum_{i=1}^m (z^{(i)} - \mu)$$

因此：

$$\frac{\partial L}{\partial \mu} = \sum_{i=1}^m \frac{\partial L}{\partial z_{\text{norm}}^{(i)}} \cdot \left(-\frac{1}{\sqrt{\sigma^2 + \epsilon}}\right) + \frac{\partial L}{\partial \sigma^2} \cdot \left(-\frac{2}{m}\sum_{i=1}^m (z^{(i)} - \mu)\right)$$

**步骤5：计算 $\frac{\partial L}{\partial z^{(i)}}$**

$z^{(i)}$ 也通过三条路径影响损失：
1. 直接通过 $z_{\text{norm}}^{(i)}$
2. 通过 $\mu$
3. 通过 $\sigma^2$

$$\frac{\partial L}{\partial z^{(i)}} = \frac{\partial L}{\partial z_{\text{norm}}^{(i)}} \cdot \frac{\partial z_{\text{norm}}^{(i)}}{\partial z^{(i)}} + \frac{\partial L}{\partial \mu} \cdot \frac{\partial \mu}{\partial z^{(i)}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial z^{(i)}}$$

其中：

$$\frac{\partial z_{\text{norm}}^{(i)}}{\partial z^{(i)}} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}$$

$$\frac{\partial \mu}{\partial z^{(i)}} = \frac{1}{m}$$

$$\frac{\partial \sigma^2}{\partial z^{(i)}} = \frac{2}{m}(z^{(i)} - \mu)$$

最终得到：

$$\frac{\partial L}{\partial z^{(i)}} = \frac{\partial L}{\partial z_{\text{norm}}^{(i)}} \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \mu} \cdot \frac{1}{m} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{2}{m}(z^{(i)} - \mu)$$


### 测试时的Batch Norm
求均值和方差时使用mini batch中m个样本进行计算，而不是整个样本数量。求一个样本的均值和方差无意义



## softmax

Softmax 是一种用于**多分类问题**的激活函数，它将神经网络的输出转换为概率分布，使得所有类别的概率之和为1。

#### 应用场景

- **二分类问题**: 使用 Sigmoid 激活函数（输出层单个神经元）
- **多分类问题**: 使用 Softmax 激活函数（输出层多个神经元，每个类别一个）

例如：手写数字识别（0-9共10类）、图像分类（猫/狗/鸟等）、文本分类等。

---

#### 正向传播 (Forward Propagation)

对于输出层的第 $j$ 个神经元，Softmax 的计算公式为：

$$a_j = \frac{e^{z_j}}{\sum_{k=1}^{C} e^{z_k}}$$

其中：
- $z_j$: 第 $j$ 个神经元的线性输出（logits）
- $C$: 类别总数
- $a_j$: 第 $j$ 个类别的预测概率

**向量化形式**（处理 m 个样本）：

假设 $Z$ 的形状为 $(C, m)$，则：

$$A = \text{softmax}(Z) = \frac{e^Z}{\sum_{k=1}^{C} e^{Z_k}}$$

**数值稳定性优化**：

直接计算 $e^{z_j}$ 可能导致数值溢出（当 $z_j$ 很大时）。解决方法是减去最大值：

$$a_j = \frac{e^{z_j - z_{\max}}}{\sum_{k=1}^{C} e^{z_k - z_{\max}}}$$

这个变换不会改变结果，因为分子分母同时除以了 $e^{z_{\max}}$。

**特性**：

1. **概率解释**: 输出值在 [0, 1] 之间，且所有类别概率之和为 1
2. **单调性**: 输入越大，输出概率越高
3. **可微性**: 处处可导，适合梯度下降优化

---

#### 损失函数：交叉熵 (Cross-Entropy Loss)

对于多分类问题，通常使用**交叉熵损失函数**。

**单样本损失**：

$$L(\hat{y}, y) = -\sum_{j=1}^{C} y_j \log(\hat{y}_j)$$

其中：
- $y_j$: 真实标签的 one-hot 编码（正确类别为1，其他为0）
- $\hat{y}_j$: 预测概率

由于 $y$ 是 one-hot 编码，只有一个位置为1，所以简化为：

$$L = -\log(\hat{y}_{\text{correct}})$$

**多样本平均损失**：

$$J = -\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{C} y_j^{(i)} \log(\hat{y}_j^{(i)})$$


#### 反向传播 (Backpropagation)

Softmax + Cross-Entropy 的反向传播有一个非常简洁的结果：


$$\frac{\partial L}{\partial z_j} = a_j - y_j$$

即：**输出层的梯度等于预测概率减去真实标签**。

**向量化形式**：

$$dZ = A - Y$$

其中：
- $dZ$: 损失对线性输出 $Z$ 的梯度 $(C, m)$
- $A$: Softmax 输出 $(C, m)$
- $Y$: 真实标签 one-hot 编码 $(C, m)$

**推导过程**：

对于第 $j$ 个输出神经元：

$$\frac{\partial L}{\partial z_j} = \sum_{k=1}^{C} \frac{\partial L}{\partial a_k} \cdot \frac{\partial a_k}{\partial z_j}$$

**情况1：$k = j$（对角线元素）**

$$\frac{\partial a_j}{\partial z_j} = a_j(1 - a_j)$$

$$\frac{\partial L}{\partial a_j} = -\frac{y_j}{a_j}$$

$$\frac{\partial L}{\partial z_j} = -\frac{y_j}{a_j} \cdot a_j(1 - a_j) = -y_j(1 - a_j) = a_j y_j - y_j$$

**情况2：$k \neq j$（非对角线元素）**

$$\frac{\partial a_k}{\partial z_j} = -a_k a_j$$

$$\frac{\partial L}{\partial a_k} = -\frac{y_k}{a_k}$$

$$\frac{\partial L}{\partial z_j} = -\frac{y_k}{a_k} \cdot (-a_k a_j) = y_k a_j$$

**合并两种情况**：

$$\frac{\partial L}{\partial z_j} = a_j y_j - y_j + \sum_{k \neq j} y_k a_j = a_j \sum_{k=1}^{C} y_k - y_j$$

由于 $\sum_{k=1}^{C} y_k = 1$（one-hot 编码），最终得到：

$$\frac{\partial L}{\partial z_j} = a_j - y_j$$

**关键结论**：

Softmax + Cross-Entropy 的组合使得梯度计算极其简洁，与二分类中 Sigmoid + Binary Cross-Entropy 的形式完全一致！

---

#### Softmax vs Sigmoid 对比

| 特性 | Softmax | Sigmoid |
|------|---------|---------|
| **适用场景** | 多分类（互斥类别） | 二分类或多标签分类 |
| **输出层神经元数** | 类别数 C | 1（二分类）或 C（多标签） |
| **输出特性** | 所有输出之和为 1 | 每个输出独立在 [0,1] |
| **概率解释** | 类别间的相对概率 | 各类别独立的概率 |
| **损失函数** | Cross-Entropy | Binary Cross-Entropy |
| **梯度形式** | $dZ = A - Y$ | $dZ = A - Y$ |

**选择建议**：

- **互斥多分类**（如：一张图只能是猫/狗/鸟之一）→ 使用 Softmax
- **多标签分类**（如：一张图可以同时包含猫和狗）→ 每个类别用独立的 Sigmoid
- **二分类** → 使用 Sigmoid（等价于两类别的 Softmax）

---

#### 注意事项

1. **数值稳定性**: 始终使用减去最大值的技巧防止溢出
2. **One-Hot 编码**: 确保标签正确转换为 one-hot 格式
3. **维度匹配**: 输出层神经元数必须等于类别数
4. **避免在输出层使用 BatchNorm**: 会破坏概率分布的性质
5. **学习率调整**: Softmax 输出对输入变化敏感，可能需要较小的学习率

---

#### 总结

Softmax 是多分类问题的标准选择，其核心优势在于：

1. **概率解释清晰**: 输出可直接理解为类别概率
2. **梯度计算简洁**: 与交叉熵配合后梯度形式简单
3. **数值稳定**: 通过减去最大值避免溢出
4. **广泛应用**: 图像分类、NLP、推荐系统等领域的基础组件

掌握 Softmax 的正反向传播推导和实现，是理解现代深度学习框架的关键基础。




### 符号
$$a_i = \frac{e^{z_i}}{S},\quad S=\sum_{k=1}^C e^{z_k}$$

### 分两种情况：$i=j$、$i\neq j$
#### ① $i = j$
$$
\begin{aligned}
\frac{\partial a_i}{\partial z_i}
&=\frac{\partial}{\partial z_i}\left(\frac{e^{z_i}}{S}\right)
=\frac{e^{z_i}\cdot S - e^{z_i}\cdot \frac{\partial S}{\partial z_i}}{S^2}\\
&\frac{\partial S}{\partial z_i}=e^{z_i}\\
&=\frac{e^{z_i}S - e^{z_i}\cdot e^{z_i}}{S^2}
=\frac{e^{z_i}}{S}-\left(\frac{e^{z_i}}{S}\right)^2\\
&=a_i - a_i^2 = a_i(1-a_i)
\end{aligned}
$$

#### ② $i \neq j$
$z_j$只在分母$S$里，分子$e^{z_i}$和$z_j$无关
$$
\begin{aligned}
\frac{\partial a_i}{\partial z_j}
&=\frac{\partial}{\partial z_j}\left(\frac{e^{z_i}}{S}\right)
=e^{z_i}\cdot\frac{0 - \frac{\partial S}{\partial z_j}}{S^2}\\
&\frac{\partial S}{\partial z_j}=e^{z_j}\\
&=-\frac{e^{z_i}e^{z_j}}{S^2}
=-a_i a_j
\end{aligned}
$$

### 汇总Softmax雅可比
$$
\frac{\partial a_i}{\partial z_j}=
\begin{cases}
a_i(1-a_i) & i=j\\
-a_i a_j & i\neq j
\end{cases}
$$

## 放回交叉熵链式推导
$$
\frac{\partial L}{\partial z_j}=\sum_{i=1}^C \frac{\partial L}{\partial a_i}\frac{\partial a_i}{\partial z_j},\quad \frac{\partial L}{\partial a_i}=-\frac{y_i}{a_i}
$$
拆成$i=j$一项 + $i\neq j$求和项：
$$
\begin{aligned}
\frac{\partial L}{\partial z_j}
&=\frac{\partial L}{\partial a_j}\cdot\frac{\partial a_j}{\partial z_j}
+\sum_{i\neq j}\frac{\partial L}{\partial a_i}\cdot\frac{\partial a_i}{\partial z_j}\\
&=-\frac{y_j}{a_j}\cdot a_j(1-a_j)
+\sum_{i\neq j}\left(-\frac{y_i}{a_i}\right)(-a_i a_j)\\
&=-y_j(1-a_j)+a_j\sum_{i\neq j}y_i\\
&=-y_j+a_j y_j+a_j\sum_{i\neq j}y_i\\
&=a_j\sum_{i=1}^C y_i - y_j
\end{aligned}
$$
one-hot：$\sum\limits_{i=1}^C y_i=1$
$$\frac{\partial L}{\partial z_j}=a_j-y_j$$