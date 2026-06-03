# 激活函数对比实验: ReLU vs Leaky ReLU vs Tanh (Softmax多分类)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import warnings

# 过滤数值警告
warnings.filterwarnings('ignore')

# 设置随机种子,保证结果可复现
np.random.seed(42)


# ==================== 1. 数据生成函数 ====================
def generate_multiclass_data(
    n_samples=1000,
    n_features=2,
    n_classes=3,
    test_size=0.2,
    random_state=42,
    cluster_std=1.0
):
    """
    使用make_blobs生成多分类数据集
    
    参数:
        n_samples: 样本总数
        n_features: 特征数量（输入维度）
        n_classes: 类别数量
        test_size: 测试集比例
        random_state: 随机种子
        cluster_std: 簇的标准差（控制数据分散程度）
    
    返回:
        X_train, X_test: 训练集和测试集特征
        y_train, y_test: 训练集和测试集标签（原始格式）
        y_train_onehot, y_test_onehot: one-hot编码后的标签
        encoder: OneHotEncoder对象
    """
    # 使用make_blobs生成多分类数据
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=cluster_std,
        random_state=random_state
    )
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 转换为one-hot编码
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
    
    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot, encoder


def visualize_dataset(X, y, title="多分类数据集"):
    """
    可视化数据集

    参数:
        X: 特征数据 (n_features, m) 或 (m, n_features)
        y: 标签数据
        title: 图表标题
    """
    # 确保X为 (m, n_features) 格式
    if X.shape[0] < X.shape[1] and X.shape[0] <= 10:
        X_plot = X.T  # 转为 (m, n_features)
    else:
        X_plot = X

    n_feat = X_plot.shape[1]

    if n_feat > 2:
        from sklearn.decomposition import PCA
        X_2d = PCA(n_components=2).fit_transform(X_plot)
        title += f" (PCA降维: {n_feat}D→2D)"
    elif n_feat < 2:
        print("警告：至少需要2维特征才能可视化，当前特征维度:", n_feat)
        return
    else:
        X_2d = X_plot

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", s=30, edgecolors='k', linewidth=0.5)
    plt.title(title, fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.colorbar(scatter, label="Class")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_data_info(X, y, y_onehot, dataset_name="数据集"):
    """
    打印数据集信息
    
    参数:
        X: 特征数据
        y: 标签数据
        y_onehot: one-hot编码后的标签
        dataset_name: 数据集名称
    """
    print(f"\n{'='*50}")
    print(f"{dataset_name}信息")
    print(f"{'='*50}")
    print(f"特征形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    print(f"One-hot形状: {y_onehot.shape}")
    print(f"类别列表: {np.unique(y)}")
    print(f"类别数量: {len(np.unique(y))}")
    print(f"各类别样本数:")
    for cls in np.unique(y):
        count = np.sum(y == cls)
        print(f"  类别 {cls}: {count} 个样本 ({count/len(y)*100:.1f}%)")


# ==================== 2. 数据加载和准备函数 ====================
def load_and_prepare_data(n_samples=1000, n_features=2, n_classes=3, test_size=0.2, random_state=42, show_plot=False, cluster_std=1.0):
    """
    加载并准备多分类数据集
    
    参数:
        n_samples: 样本总数
        n_features: 特征数量
        n_classes: 类别数量
        test_size: 测试集比例
        random_state: 随机种子
        show_plot: 是否显示数据可视化图像（默认False，仅保存文件）
        cluster_std: 簇的标准差（控制数据分散程度）
    
    返回:
        X_train, Y_train, X_test, Y_test: 训练集和测试集（已转置）
        y_train_raw, y_test_raw: 原始标签
        encoder: OneHotEncoder对象
    """
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, y_train_onehot, y_test_onehot, encoder = generate_multiclass_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        test_size=test_size,
        random_state=random_state,
        cluster_std=cluster_std
    )

    # 打印数据集信息
    print_data_info(X_train_raw, y_train_raw, y_train_onehot, "训练集")
    print_data_info(X_test_raw, y_test_raw, y_test_onehot, "测试集")

    # 可视化数据集（可选）
    if show_plot:
        X_full = np.vstack([X_train_raw, X_test_raw])
        y_full = np.concatenate([y_train_raw, y_test_raw])
        visualize_dataset(X_full, y_full, title=f"多分类数据集（{n_classes}类）")

    # 转置为 (特征数, 样本数)
    X_train = X_train_raw.T  # (2, 800)
    Y_train = y_train_onehot.T  # (3, 800)
    X_test = X_test_raw.T  # (2, 200)
    Y_test = y_test_onehot.T  # (3, 200)

    n_features, m_train = X_train.shape
    n_classes = Y_train.shape[0]
    print(f"\n训练集: X{X_train.shape}, Y{Y_train.shape}")
    print(f"测试集: X{X_test.shape}, Y{Y_test.shape}")
    print(f"特征数: {n_features}, 类别数: {n_classes}")
    
    return X_train, Y_train, X_test, Y_test, y_train_raw, y_test_raw, encoder


# ==================== 3. 激活函数定义 ====================
def softmax(z):
    """Softmax激活函数（输出层）- 数值稳定版本"""
    # 减去最大值以防止数值溢出
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def sigmoid(z):
    """Sigmoid激活函数（仅用于二分类对比）"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    """Sigmoid的导数"""
    s = sigmoid(z)
    return s * (1 - s)

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


# ==================== 4. 神经网络核心函数 ====================
def forward_propagation(X, w1, b1, w2, b2, activation='tanh', use_softmax=False):
    """前向传播
    
    参数:
        X: 输入数据 (n_features, m)
        w1, b1: 隐藏层权重和偏置
        w2, b2: 输出层权重和偏置
        activation: 隐藏层激活函数类型
        use_softmax: 是否使用softmax作为输出层（多分类）
    
    返回:
        z1, a1, z2, a2: 各层的线性输出和激活输出
    """
    act_func = ACTIVATION_FUNCTIONS[activation]['func']
    
    # 隐藏层
    z1 = np.dot(w1.T, X) + b1
    a1 = act_func(z1)

    # 输出层
    z2 = np.dot(w2.T, a1) + b2
    
    if use_softmax:
        # 多分类：使用softmax
        a2 = softmax(z2)
    else:
        # 二分类：使用sigmoid
        a2 = sigmoid(z2)

    return z1, a1, z2, a2

def cross_entropy_loss(A, Y):
    """交叉熵损失函数（适用于softmax多分类）
    
    参数:
        A: 预测概率 (n_classes, m)
        Y: 真实标签one-hot编码 (n_classes, m)
    
    返回:
        loss: 平均交叉熵损失
    """
    m = Y.shape[1]
    epsilon = 1e-8
    # 防止log(0)
    loss = -(1 / m) * np.sum(Y * np.log(A + epsilon))
    return loss

def logistic_loss(A, Y):
    """逻辑回归损失函数（仅用于二分类对比）"""
    m = Y.shape[1]
    epsilon = 1e-8
    return -(1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))

def backward_propagation(X, Y, W2, z1, a1, z2, a2, activation='tanh', use_softmax=False):
    """反向传播
    
    参数:
        X: 输入数据 (n_features, m)
        Y: 真实标签 (n_classes, m)
        W2: 输出层权重
        z1, a1: 隐藏层的线性输出和激活输出
        z2, a2: 输出层的线性输出和激活输出
        activation: 隐藏层激活函数类型
        use_softmax: 是否使用softmax
    
    返回:
        dw1, db1, dw2, db2: 各参数的梯度
    """
    act_deriv = ACTIVATION_FUNCTIONS[activation]['deriv']
    m = X.shape[1]

    if use_softmax:
        # Softmax + Cross Entropy 的梯度简化形式
        dz2 = a2 - Y  # (n_classes, m)
    else:
        # Sigmoid + Binary Cross Entropy
        dz2 = a2 - Y  # (1, m)
    
    dw2 = (1 / m) * np.dot(a1, dz2.T)  # (h, n_classes)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)  # (n_classes, 1)
    
    act_derivative = act_deriv(a1)
    dz1 = np.dot(W2, dz2) * act_derivative  # (h, m)

    dw1 = (1 / m) * np.dot(X, dz1.T)  # (n_features, h)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)  # (h, 1)

    return dw1, db1, dw2, db2

def predict(X, w1, b1, w2, b2, activation='tanh', use_softmax=False):
    """预测函数
    
    参数:
        X: 输入数据
        w1, b1, w2, b2: 网络参数
        activation: 隐藏层激活函数
        use_softmax: 是否使用softmax输出
    
    返回:
        predictions: 预测的类别标签（非one-hot）
    """
    _, _, _, A = forward_propagation(X, w1, b1, w2, b2, activation=activation, use_softmax=use_softmax)
    
    if use_softmax:
        # 多分类：取概率最大的类别
        return np.argmax(A, axis=0)
    else:
        # 二分类：阈值判断
        return (A >= 0.5).astype(int)

def accuracy(Y_pre, Y_true):
    """准确率计算
    
    参数:
        Y_pre: 预测标签 (m,) 或 (1, m)
        Y_true: 真实标签 (m,) 或 (n_classes, m) one-hot
    
    返回:
        acc: 准确率
    """
    # 如果Y_true是one-hot编码，转换为类别索引
    if Y_true.ndim > 1 and Y_true.shape[0] > 1:
        Y_true_flat = np.argmax(Y_true, axis=0)
    else:
        Y_true_flat = Y_true.flatten()
    
    # 确保Y_pre是一维数组
    Y_pre_flat = Y_pre.flatten()
    
    return np.mean(Y_pre_flat == Y_true_flat)

def train_neural_network(X_train, Y_train, X_test, Y_test, activation_type, loop=8000, learn_rate=0.4, h=4, use_softmax=True):
    """训练神经网络并返回历史记录
    
    参数:
        X_train, Y_train: 训练集
        X_test, Y_test: 测试集
        activation_type: 隐藏层激活函数类型
        loop: 训练迭代次数
        learn_rate: 学习率
        h: 隐藏层神经元数量
        use_softmax: 是否使用softmax（多分类）
    
    返回:
        包含训练结果和历史的字典
    """
    np.random.seed(42)  # 确保每次训练使用相同的初始权重
    
    n_features = X_train.shape[0]
    n_classes = Y_train.shape[0] if use_softmax else 1
    
    # 初始化参数
    w1 = np.random.randn(n_features, h) * 0.01
    b1 = np.zeros((h, 1))
    w2 = np.random.randn(h, n_classes) * 0.01
    b2 = np.zeros((n_classes, 1))
    
    loss_history = []
    train_acc_history = []
    test_acc_history = []
    
    print(f"\n{'='*60}")
    print(f"训练激活函数: {activation_type} {'(Softmax多分类)' if use_softmax else '(Sigmoid二分类)'}")
    print(f"{'='*60}")
    
    for i in range(loop):
        z1, a1, z2, a2 = forward_propagation(X_train, w1, b1, w2, b2, 
                                              activation=activation_type, 
                                              use_softmax=use_softmax)
        
        # 选择损失函数
        if use_softmax:
            loss = cross_entropy_loss(a2, Y_train)
        else:
            loss = logistic_loss(a2, Y_train)
        
        dw1, db1, dw2, db2 = backward_propagation(X_train, Y_train, w2, z1, a1, z2, a2, 
                                                   activation=activation_type,
                                                   use_softmax=use_softmax)
        
        w1 = w1 - learn_rate * dw1
        b1 = b1 - learn_rate * db1
        w2 = w2 - learn_rate * dw2
        b2 = b2 - learn_rate * db2
        
        if i % 100 == 0:
            loss_history.append(loss)
            
            Y_pred_train = predict(X_train, w1, b1, w2, b2, activation=activation_type, use_softmax=use_softmax)
            train_acc = accuracy(Y_pred_train, Y_train)
            train_acc_history.append(train_acc)
            
            Y_pred_test = predict(X_test, w1, b1, w2, b2, activation=activation_type, use_softmax=use_softmax)
            test_acc = accuracy(Y_pred_test, Y_test)
            test_acc_history.append(test_acc)
            
            if i % 1000 == 0:
                print(f"Iteration {i:5d}, Loss: {loss:.6f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # 最终准确率
    Y_pred_test = predict(X_test, w1, b1, w2, b2, activation=activation_type, use_softmax=use_softmax)
    final_test_acc = accuracy(Y_pred_test, Y_test)
    Y_pred_train = predict(X_train, w1, b1, w2, b2, activation=activation_type, use_softmax=use_softmax)
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

def plot_decision_boundary(X, Y, w1, b1, w2, b2, ax, activation='tanh', use_softmax=True, title='Decision Boundary'):
    """绘制决策边界

    参数:
        X: 输入数据 (n_features, m)
        Y: 标签数据（one-hot或原始）
        w1, b1, w2, b2: 网络参数
        ax: matplotlib轴对象
        activation: 隐藏层激活函数
        use_softmax: 是否使用softmax
        title: 图表标题
    """
    n_features = X.shape[0]

    if n_features > 2:
        # 高维：PCA降维到2D用于可视化
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2).fit(X.T)
        X_2d = pca.transform(X.T).T  # (2, m)
    else:
        X_2d = X

    x_min, x_max = X_2d[0, :].min() - 0.5, X_2d[0, :].max() + 0.5
    y_min, y_max = X_2d[1, :].min() - 0.5, X_2d[1, :].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    grid_2d = np.c_[xx.ravel(), yy.ravel()].T  # (2, grid_size)

    if n_features > 2:
        # 将2D网格点逆变换回原始特征空间
        grid_points = pca.inverse_transform(grid_2d.T).T  # (n_features, grid_size)
    else:
        grid_points = grid_2d

    predictions = predict(grid_points, w1, b1, w2, b2, activation=activation, use_softmax=use_softmax)
    predictions = predictions.reshape(xx.shape)

    # 如果是多分类，使用不同的colormap
    if use_softmax and predictions.max() > 1:
        cmap = plt.cm.RdYlBu
    else:
        cmap = plt.cm.RdBu

    ax.contourf(xx, yy, predictions, cmap=cmap, alpha=0.3)

    # 将Y转换为适合可视化的格式
    if Y.ndim > 1 and Y.shape[0] > 1:
        Y_vis = np.argmax(Y, axis=0)
    else:
        Y_vis = Y[0, :] if Y.ndim > 1 else Y

    ax.scatter(X_2d[0, :], X_2d[1, :], c=Y_vis, cmap=cmap, edgecolors='k', s=30)
    feat_label = "PCA " if n_features > 2 else ""
    ax.set_xlabel(f'{feat_label}Feature 1', fontsize=10)
    ax.set_ylabel(f'{feat_label}Feature 2', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


# ==================== 5. 训练所有激活函数 ====================
def train_all_activations(X_train, Y_train, X_test, Y_test, activation_types=None, loop_count=20000, learn_rate=0.4, h=8):
    """
    训练所有指定的激活函数
    
    参数:
        X_train, Y_train: 训练集
        X_test, Y_test: 测试集
        activation_types: 要训练的激活函数类型列表
        loop_count: 训练迭代次数
        learn_rate: 学习率
        h: 隐藏层神经元数量
    
    返回:
        results: 包含所有激活函数训练结果的字典
    """
    if activation_types is None:
        activation_types = ['relu', 'leaky_relu', 'tanh', 'sigmoid']
    
    results = {}
    
    for act_type in activation_types:
        results[act_type] = train_neural_network(
            X_train, Y_train, X_test, Y_test, 
            activation_type=act_type,
            loop=loop_count,
            learn_rate=learn_rate,
            h=h
        )
    
    return results


# ==================== 6. 可视化结果 ====================
def visualize_results(results, activation_types=None, loop_count=20000, X_train=None, Y_train=None):
    """
    可视化所有激活函数的训练结果和决策边界
    
    参数:
        results: 训练结果字典
        activation_types: 激活函数类型列表
        loop_count: 训练迭代次数
        X_train, Y_train: 训练集数据（用于绘制决策边界）
    """
    if activation_types is None:
        activation_types = list(results.keys())
    
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
        if X_train is not None and Y_train is not None:
            ax2 = axes[idx, 1]
            plot_decision_boundary(
                X_train, Y_train, 
                result['w1'], result['b1'], result['w2'], result['b2'],
                ax2, 
                activation=act_type,
                use_softmax=True,  # 使用softmax多分类
                title=f'{act_type.upper()} - Decision Boundary'
            )

    plt.tight_layout()
    plt.savefig('activation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print("对比图像已保存为 activation_comparison.png")
    print(f"{'='*60}")
    # plt.show()  # 在终端环境中注释掉，避免阻塞


# ==================== 7. 主函数 ====================
def main(n_features=4):
    """主函数：执行完整的激活函数对比实验

    参数:
        n_features: 特征数量（>=2时使用PCA降维可视化，=2时直接绘制）
    """
    # 加载和准备数据（使用make_blobs生成）
    X_train, Y_train, X_test, Y_test, y_train_raw, y_test_raw, encoder = load_and_prepare_data(
        n_samples=2000,
        n_features=n_features,
        n_classes=8,
        test_size=0.2,
        random_state=60,
        show_plot=False,  # 显示数据可视化图像
        cluster_std=1.0  # 控制数据簇的分散程度
    )
    loop_count = 40000
    # 训练所有激活函数（使用softmax多分类）
    activation_types = ['relu', 'sigmoid']  # sigmoid不适用于多分类
    results = train_all_activations(
        X_train, Y_train, X_test, Y_test,
        activation_types=activation_types,
        loop_count=loop_count,
        learn_rate=0.2,
        h=4
    )
    
    # 可视化结果
    visualize_results(
        results,
        activation_types=activation_types,
        loop_count=loop_count,
        X_train=X_train,
        Y_train=Y_train
    )
    
    return results


# ==================== 8. 程序入口 ====================
if __name__ == "__main__":
    results = main()
