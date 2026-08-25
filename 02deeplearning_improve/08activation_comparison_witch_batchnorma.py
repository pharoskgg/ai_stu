# 激活函数对比实验: ReLU vs Leaky ReLU vs Tanh (Mini-batch SGD + Batch Normalization)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import warnings

# 过滤数值警告
warnings.filterwarnings('ignore')

# 设置随机种子,保证结果可复现
np.random.seed(42)

# ==================== 1. 生成数据集 ====================
X, Y = make_moons(n_samples=2000, noise=0.2, random_state=42)

# 划分训练集和测试集 (80%训练, 20%测试)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

# 转置为 (特征数, 样本数)
X_train = X_train.T  # (2, 480)
Y_train = Y_train.reshape(1, -1)  # (1, 480)
X_test = X_test.T  # (2, 120)
Y_test = Y_test.reshape(1, -1)  # (1, 120)

n_features, m_train = X_train.shape
print(f"训练集: X{X_train.shape}, Y{Y_train.shape}")
print(f"测试集: X{X_test.shape}, Y{Y_test.shape}")


# ==================== 2. 激活函数定义 ====================
def sigmoid(z):
    """Sigmoid激活函数（输出层）"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    """Sigmoid的导数"""
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    """ReLU激活函数"""
    return np.maximum(0, z)

def relu_derivative(a):
    """ReLU的导数"""
    return (a > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU激活函数"""
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(a, alpha=0.4):
    """Leaky ReLU的导数"""
    return np.where(a > 0, 1.0, alpha)

def tanh(z):
    """Tanh激活函数"""
    return np.tanh(z)

def tanh_derivative(a):
    """Tanh的导数: 1 - a^2"""
    return 1 - a ** 2

# 激活函数映射表
ACTIVATION_FUNCTIONS = {
    'relu': {'func': relu, 'deriv': relu_derivative},
    'leaky_relu': {'func': leaky_relu, 'deriv': leaky_relu_derivative},
    'tanh': {'func': tanh, 'deriv': tanh_derivative}
}

# ==================== 3. Batch Normalization 辅助函数 ====================
def batch_norm_forward(z, gamma, beta, eps=1e-8, training=True,
                       running_mean=None, running_var=None, momentum=0.9):
    """Batch Normalization 前向传播。

    训练时用当前 batch 统计量并更新 running 统计量（动量 0.9，与 KSNet 一致）；
    推理时用 running 统计量。
    """
    if training:
        mu = np.mean(z, axis=1, keepdims=True)
        var = np.var(z, axis=1, keepdims=True)
        sigma = np.sqrt(var + eps)
        z_norm = (z - mu) / sigma

        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var

        cache = (z, mu, var, sigma, z_norm)
    else:
        sigma = np.sqrt(running_var + eps)
        z_norm = (z - running_mean) / sigma
        cache = None

    z_out = gamma * z_norm + beta
    return z_out, cache, running_mean, running_var

def batch_norm_backward(dz_out, cache, gamma, eps=1e-8):
    """
    Batch Normalization 反向传播（修正版标准公式）
    """
    z, mu, var, sigma, z_norm = cache
    m = z.shape[1]

    # 梯度计算
    dgamma = np.sum(dz_out * z_norm, axis=1, keepdims=True)
    dbeta = np.sum(dz_out, axis=1, keepdims=True)

    dz_norm = dz_out * gamma
    dvar = np.sum(dz_norm * (z - mu) * (-0.5) * (var + eps)**(-3/2), axis=1, keepdims=True)
    dmu = np.sum(dz_norm * (-1 / sigma), axis=1, keepdims=True) + dvar * np.mean(-2 * (z - mu), axis=1, keepdims=True)

    dz = (dz_norm / sigma) + (dmu / m) + (dvar * 2 * (z - mu) / m)
    return dz, dgamma, dbeta

# ==================== 4. 神经网络核心函数 ====================
def forward_propagation(X, w1, b1, w2, b2, gamma1, beta1, activation='tanh',
                        training=True, running_mean=None, running_var=None):
    act_func = ACTIVATION_FUNCTIONS[activation]['func']

    # 隐藏层
    z1 = np.dot(w1.T, X) + b1
    z1_bn, bn_cache1, running_mean, running_var = batch_norm_forward(
        z1, gamma1, beta1, training=training,
        running_mean=running_mean, running_var=running_var)
    a1 = act_func(z1_bn)

    # 输出层
    z2 = np.dot(w2.T, a1) + b2
    a2 = sigmoid(z2)

    return z1, a1, a2, bn_cache1, running_mean, running_var

def logistic_loss(A, Y):
    """逻辑回归损失函数"""
    m = Y.shape[1]
    epsilon = 1e-8
    return -(1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))

def backward_propagation(X, Y, W2, a1, a2, bn_cache1, gamma1, activation='tanh'):
    act_deriv = ACTIVATION_FUNCTIONS[activation]['deriv']
    m = X.shape[1]

    # 统一为均值约定：dz2 直接带 1/m，后续各层（含 BN 的 gamma/beta）不再重复除 m
    dz2 = (a2 - Y) / m
    dw2 = np.dot(a1, dz2.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    # 隐藏层梯度
    da1 = np.dot(W2, dz2)
    dz1_bn = da1 * act_deriv(a1)

    # BN 反向
    dz1, dgamma1, dbeta1 = batch_norm_backward(dz1_bn, bn_cache1, gamma1)

    dw1 = np.dot(X, dz1.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2, dgamma1, dbeta1

def predict(X, w1, b1, w2, b2, gamma1, beta1, running_mean, running_var, activation='tanh'):
    _, _, A, _, _, _ = forward_propagation(X, w1, b1, w2, b2, gamma1, beta1,
                                           activation=activation, training=False,
                                           running_mean=running_mean, running_var=running_var)
    return (A >= 0.5).astype(int)

def accuracy(Y_pre, Y_true):
    return np.mean(Y_pre == Y_true)

def generate_mini_batches(X, Y, mini_batch_size, seed):
    """将数据随机打乱后按 mini_batch_size 切分为多个 mini-batch

    一轮(epoch)内每个样本只出现一次，抽过的不再放回；
    剩余数量不足 mini_batch_size 时返回剩余数据；
    下一轮调用时重新打乱，样本可再次被抽中。
    """
    np.random.seed(seed)
    m = X.shape[1]
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]

    mini_batches = []
    num_complete = m // mini_batch_size
    for k in range(num_complete):
        start = k * mini_batch_size
        end = start + mini_batch_size
        mini_batches.append((X_shuffled[:, start:end], Y_shuffled[:, start:end]))

    # 剩余不足 mini_batch_size 的样本
    if m % mini_batch_size != 0:
        start = num_complete * mini_batch_size
        mini_batches.append((X_shuffled[:, start:], Y_shuffled[:, start:]))

    return mini_batches

def train_neural_network(X_train, Y_train, X_test, Y_test, activation_type,
                         num_epochs=3000, learn_rate=0.05, h=8, mini_batch_size=64):
    """训练神经网络并返回历史记录（Mini-batch SGD, epoch遍历式）"""
    np.random.seed(42)  # 确保每次训练使用相同的初始权重

    n_features = X_train.shape[0]

    # 初始化参数
    w1 = np.random.randn(n_features, h) * 0.01
    b1 = np.zeros((h, 1))
    w2 = np.random.randn(h, 1) * 0.01
    b2 = np.zeros((1, 1))
    gamma1 = np.ones((h, 1))
    beta1 = np.zeros((h, 1))
    running_mean = np.zeros((h, 1))
    running_var = np.ones((h, 1))

    loss_history = []
    train_acc_history = []
    test_acc_history = []
    step = 0

    print(f"\n{'='*60}")
    print(f"训练激活函数: {activation_type}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        # 每个 epoch 重新打乱并切分 mini-batch
        mini_batches = generate_mini_batches(X_train, Y_train, mini_batch_size, seed=epoch)

        for X_batch, Y_batch in mini_batches:
            z1, a1, a2, bn_cache1, running_mean, running_var = forward_propagation(
                X_batch, w1, b1, w2, b2, gamma1, beta1, activation_type,
                training=True, running_mean=running_mean, running_var=running_var)
            loss = logistic_loss(a2, Y_batch)

            dw1, db1, dw2, db2, dgamma1, dbeta1 = backward_propagation(
                X_batch, Y_batch, w2, a1, a2, bn_cache1, gamma1, activation_type)

            # 更新
            w1 = w1 - learn_rate * dw1
            b1 = b1 - learn_rate * db1
            w2 = w2 - learn_rate * dw2
            b2 = b2 - learn_rate * db2
            gamma1 -= learn_rate * dgamma1
            beta1 -= learn_rate * dbeta1

            if step % 50 == 0:
                loss_history.append(loss)
                Y_pred_train = predict(X_train, w1, b1, w2, b2, gamma1, beta1,
                                       running_mean, running_var, activation_type)
                train_acc_history.append(accuracy(Y_pred_train, Y_train))
                Y_pred_test = predict(X_test, w1, b1, w2, b2, gamma1, beta1,
                                      running_mean, running_var, activation_type)
                test_acc_history.append(accuracy(Y_pred_test, Y_test))

                if step % 1000 == 0:
                    print(f"Step {step:5d} | Loss {loss:.6f} | Train {train_acc_history[-1]:.4f} | Test {test_acc_history[-1]:.4f}")

            step += 1

    # 最终结果
    final_train_acc = accuracy(predict(X_train, w1, b1, w2, b2, gamma1, beta1,
                                       running_mean, running_var, activation_type), Y_train)
    final_test_acc = accuracy(predict(X_test, w1, b1, w2, b2, gamma1, beta1,
                                      running_mean, running_var, activation_type), Y_test)

    print(f"\n最终 → 训练: {final_train_acc*100:.2f}% | 测试: {final_test_acc*100:.2f}%")

    return {
        'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,
        'gamma1': gamma1, 'beta1': beta1,
        'running_mean': running_mean, 'running_var': running_var,
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'test_acc_history': test_acc_history,
        'final_train_acc': final_train_acc,
        'final_test_acc': final_test_acc
    }

def plot_decision_boundary(X, Y, w1, b1, w2, b2, gamma1, beta1, running_mean, running_var, ax, activation='tanh', title='Decision Boundary'):
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    predictions = predict(grid_points, w1, b1, w2, b2, gamma1, beta1,
                          running_mean, running_var, activation=activation)
    predictions = predictions.reshape(xx.shape)

    ax.contourf(xx, yy, predictions, cmap=plt.cm.RdBu, alpha=0.3)
    ax.scatter(X[0, :], X[1, :], c=Y[0, :], cmap=plt.cm.RdBu, edgecolors='k', s=20)
    ax.set_title(title, fontweight='bold')

# ==================== 5. 训练三种激活函数 ====================
activation_types = ['relu', 'leaky_relu', 'tanh']
results = {}

for act_type in activation_types:
    results[act_type] = train_neural_network(
        X_train, Y_train, X_test, Y_test,
        activation_type=act_type,
        num_epochs=6000,
        learn_rate=0.1,
        h=4,
        mini_batch_size=512
    )

# ==================== 6. 可视化对比 ====================
fig, axes = plt.subplots(len(activation_types), 2, figsize=(14, 12), squeeze=False)

for idx, act_type in enumerate(activation_types):
    res = results[act_type]
    iters = np.arange(len(res['loss_history'])) * 50

    # 训练曲线
    ax1 = axes[idx, 0]
    ax1.plot(iters, res['loss_history'], 'r', label='Loss')
    ax1.plot(iters, res['train_acc_history'], 'b', label='Train Acc')
    ax1.plot(iters, res['test_acc_history'], 'g', label='Test Acc')
    ax1.set_title(f'{act_type.upper()} | Train:{res["final_train_acc"]*100:.1f}% Test:{res["final_test_acc"]*100:.1f}%')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 决策边界
    ax2 = axes[idx, 1]
    plot_decision_boundary(X_train, Y_train, res['w1'], res['b1'], res['w2'], res['b2'],
                            res['gamma1'], res['beta1'], res['running_mean'], res['running_var'],
                            ax2, act_type, f'{act_type.upper()} 决策边界')

plt.tight_layout()
plt.savefig('08activation_comparison_with_batchnormal.png', dpi=150, bbox_inches='tight')