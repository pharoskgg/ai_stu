# 激活函数对比实验: ReLU vs Leaky ReLU vs Tanh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import warnings

# 过滤数值警告
warnings.filterwarnings('ignore')

# 设置随机种子,保证结果可复现
np.random.seed(42)

# ==================== 1. 生成数据集 ====================
X, Y = make_moons(n_samples=600, noise=0.2, random_state=42)

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
    'tanh': {'func': tanh, 'deriv': tanh_derivative},
    'sigmoid': {'func': sigmoid, 'deriv': sigmoid_derivative}
}


# ==================== 3. 神经网络核心函数 ====================
def forward_propagation(X, w1, b1, w2, b2, activation='tanh', keep_prob=1.0):
    """前向传播
    keep_prob=1.0 表示不使用dropout（测试时默认关闭）
    """
    act_func = ACTIVATION_FUNCTIONS[activation]['func']

    # 隐藏层
    z1 = np.dot(w1.T, X) + b1
    a1 = act_func(z1)

    # Dropout（仅隐藏层，keep_prob < 1 时启用）
    D1 = None
    if keep_prob < 1.0:
        # D1 = np.random.rand(a1.shape[0], a1.shape[1]) < keep_prob
        # 与框架的实现生成掩码保持一致
        D1 = (np.random.rand(a1.shape[1], a1.shape[0]) >= (1 - keep_prob)).T
        a1 *= D1
        a1 /= keep_prob

    # 输出层（不做dropout）
    z2 = np.dot(w2.T, a1) + b2
    a2 = sigmoid(z2)

    return z1, a1, a2, D1

def logistic_loss(A, Y):
    """逻辑回归损失函数"""
    m = Y.shape[1]
    epsilon = 1e-8
    return -(1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))

def backward_propagation(X, Y, W2, z1, a1, a2, activation='tanh', D1=None, keep_prob=1.0):
    """反向传播
    D1: 前向传播生成的dropout掩码，用于屏蔽被杀死神经元的梯度
    """
    act_deriv = ACTIVATION_FUNCTIONS[activation]['deriv']
    m = X.shape[1]

    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(a1, dz2.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

    act_derivative = act_deriv(a1)
    dz1 = np.dot(W2, dz2) * act_derivative

    # Dropout梯度：被杀死的神经元梯度归零
    if D1 is not None:
        dz1 *= D1 / keep_prob

    dw1 = (1 / m) * np.dot(X, dz1.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2

def predict(X, w1, b1, w2, b2, activation='tanh'):
    """预测函数（关闭dropout）"""
    _, _, A, _ = forward_propagation(X, w1, b1, w2, b2, activation=activation, keep_prob=1.0)
    return (A >= 0.5).astype(int)

def accuracy(Y_pre, Y_true):
    """准确率计算"""
    return np.mean(Y_pre == Y_true)

def train_neural_network(X_train, Y_train, X_test, Y_test, activation_type, loop=8000, learn_rate=0.4, h=4):
    """训练神经网络并返回历史记录"""
    np.random.seed(42)  # 确保每次训练使用相同的初始权重
    
    n_features = X_train.shape[0]
    
    # 初始化参数
    w1 = np.random.randn(n_features, h) * 0.01
    b1 = np.zeros((h, 1))
    w2 = np.random.randn(h, 1) * 0.01
    b2 = np.zeros((1, 1))
    
    loss_history = []
    train_acc_history = []
    test_acc_history = []
    
    print(f"\n{'='*60}")
    print(f"训练激活函数: {activation_type}")
    print(f"{'='*60}")
    
    for i in range(loop):
        z1, a1, a2, D1 = forward_propagation(X_train, w1, b1, w2, b2, activation=activation_type, keep_prob=0.8)
        loss = logistic_loss(a2, Y_train)

        dw1, db1, dw2, db2 = backward_propagation(X_train, Y_train, w2, z1, a1, a2, activation=activation_type, D1=D1, keep_prob=0.8)
        
        w1 = w1 - learn_rate * dw1
        b1 = b1 - learn_rate * db1
        w2 = w2 - learn_rate * dw2
        b2 = b2 - learn_rate * db2
        
        if i % 100 == 0:
            loss_history.append(loss)
            
            Y_pred_train = predict(X_train, w1, b1, w2, b2, activation=activation_type)
            train_acc = accuracy(Y_pred_train, Y_train)
            train_acc_history.append(train_acc)
            
            Y_pred_test = predict(X_test, w1, b1, w2, b2, activation=activation_type)
            test_acc = accuracy(Y_pred_test, Y_test)
            test_acc_history.append(test_acc)
            
            if i % 1000 == 0:
                print(f"Iteration {i:5d}, Loss: {loss:.6f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # 最终准确率
    Y_pred_test = predict(X_test, w1, b1, w2, b2, activation=activation_type)
    final_test_acc = accuracy(Y_pred_test, Y_test)
    Y_pred_train = predict(X_train, w1, b1, w2, b2, activation=activation_type)
    final_train_acc = accuracy(Y_pred_train, Y_train)
    
    print(f"\n最终结果 - 训练集准确率: {final_train_acc*100:.2f}%, 测试集准确率: {final_test_acc*100:.2f}%")
    
    return {
        'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'test_acc_history': test_acc_history,
        'final_train_acc': final_train_acc,
        'final_test_acc': final_test_acc
    }

def plot_decision_boundary(X, Y, w1, b1, w2, b2, ax, activation='tanh', title='Decision Boundary'):
    """绘制决策边界"""
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    predictions = predict(grid_points, w1, b1, w2, b2, activation=activation)
    predictions = predictions.reshape(xx.shape)
    
    ax.contourf(xx, yy, predictions, cmap=plt.cm.RdBu, alpha=0.3)
    ax.scatter(X[0, :], X[1, :], c=Y[0, :], cmap=plt.cm.RdBu, edgecolors='k', s=30)
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

loop_count = 20000
lear_rate=0.4
# ==================== 4. 训练三种激活函数 ====================
activation_types = ['relu', 'leaky_relu', 'tanh', 'sigmoid']
results = {}

for act_type in activation_types:
    results[act_type] = train_neural_network(
        X_train, Y_train, X_test, Y_test, 
        activation_type=act_type,
        loop=loop_count,
        learn_rate=lear_rate,
        h=8
    )


# ==================== 5. 可视化对比 ====================
fig, axes = plt.subplots(len(activation_types), 2, figsize=(16, 18))

# 根据实际训练轮数生成iterations

iterations = list(range(0, loop_count, 100))

for idx, act_type in enumerate(activation_types):
    result = results[act_type]
    
    # 左图: Loss和准确率曲线
    ax1 = axes[idx, 0]
    ax1.plot(iterations, result['loss_history'], 'r-', linewidth=2, label='Loss')
    ax1.plot(iterations, result['train_acc_history'], 'b-', linewidth=2, label='Train Accuracy')
    ax1.plot(iterations, result['test_acc_history'], 'g-', linewidth=2, label='Test Accuracy')
    ax1.set_xlabel('Iteration', fontsize=10)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.set_title(f'{act_type.upper()} - Training Progress\n' + 
                  f'Train Acc: {result["final_train_acc"]*100:.2f}%, Test Acc: {result["final_test_acc"]*100:.2f}%', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, loop_count])
    
    # 右图: 决策边界
    ax2 = axes[idx, 1]
    plot_decision_boundary(
        X_train, Y_train, 
        result['w1'], result['b1'], result['w2'], result['b2'],
        ax2, 
        activation=act_type,
        title=f'{act_type.upper()} - Decision Boundary'
    )

plt.tight_layout()
# plt.show()  # 在终端环境中注释掉，避免阻塞
plt.savefig('activation_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*60}")
print("对比图像已保存为 activation_comparison.png")
print(f"{'='*60}")
# plt.show()
