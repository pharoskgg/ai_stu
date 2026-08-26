import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split


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


def generate_multiclass_dataset(
    n_samples=2000,
    n_features=4,
    n_classes=8,
    test_size=0.2,
    random_state=60,
    cluster_std=1.0,
):
    """生成 ``make_blobs`` 多分类数据，并将标签转换为 One-hot。

    数据使用 KSNet 的行主序布局：特征形状为 ``(样本数, 特征数)``，
    标签形状为 ``(样本数, 类别数)``。

    :return: ``(X_train, Y_train), (X_test, Y_test)``
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 每个类别索引会选取单位矩阵中对应的一行，从而得到 One-hot 标签。
    one_hot_lookup = np.eye(n_classes, dtype=np.float64)
    Y_train = one_hot_lookup[y_train]
    Y_test = one_hot_lookup[y_test]
    return (X_train, Y_train), (X_test, Y_test)
