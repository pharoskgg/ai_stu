from sklearn.datasets import make_moons


def generate_moons_dataset(n_samples=300, noise=0.2, random_state=42, train_ratio=0.8):
    """
    生成月牙形二分类数据集，并按比例切分为训练集和测试集
    :param n_samples: 总样本数
    :param noise: 噪声程度
    :param random_state: 随机种子，保证结果可复现
    :param train_ratio: 训练集占比
    :return: (X_train, Y_train), (X_test, Y_test)
    """
    X, Y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    split_idx = int(train_ratio * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    return (X_train, Y_train), (X_test, Y_test)
