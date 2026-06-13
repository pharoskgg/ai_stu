"""
MNIST数据集加载器 - 支持Mini-Batch方式获取数据

功能说明:
1. 从data目录加载已解压的MNIST图片数据集
2. 支持以mini-batch方式获取训练数据
3. 每轮(epoch)中确保不重复获取相同数据
4. 自动打乱数据顺序
5. 支持归一化和one-hot编码转换

目录结构:
data/
├── train/
│   ├── images/  (60000张 .png 图片)
│   └── labels.txt  (标签文件，每行一个数字)
└── test/
    ├── images/  (10000张 .png 图片)
    └── labels.txt  (标签文件)
"""

import numpy as np
import os
from PIL import Image
import glob


class MNISTDataLoader:
    """MNIST数据集加载器，支持mini-batch获取"""
    
    def __init__(self, data_dir=None, batch_size=64, shuffle=True, normalize=True):
        """
        初始化数据加载器
        
        参数:
            data_dir: 数据集根目录路径（默认为当前脚本所在目录下的data文件夹）
            batch_size: mini-batch大小
            shuffle: 是否在每个epoch开始时打乱数据
            normalize: 是否将像素值归一化到[0, 1]区间
        """
        if data_dir is None:
            # 默认使用当前脚本所在目录下的data文件夹
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        
        # 加载训练集和测试集
        self.X_train, self.y_train = self._load_dataset("train")
        self.X_test, self.y_test = self._load_dataset("test")
        
        # 训练集相关状态
        self.train_indices = None  # 当前epoch的索引顺序
        self.current_index = 0  # 当前读取位置
        self.epoch_count = 0  # 已完成的epoch数
        
        print(f"✓ 训练集加载完成: {self.X_train.shape[0]} 个样本")
        print(f"✓ 测试集加载完成: {self.X_test.shape[0]} 个样本")
        print(f"✓ Batch size: {batch_size}, Shuffle: {shuffle}, Normalize: {normalize}")
    
    def _load_dataset(self, split="train"):
        """
        加载指定数据集（训练集或测试集）
        
        参数:
            split: 'train' 或 'test'
        
        返回:
            X: 图像数据数组，形状为 (n_samples, 784)，值为 [0, 1] 或 [0, 255]
            y: 标签数组，形状为 (n_samples,)
        """
        split_dir = os.path.join(self.data_dir, split)
        images_dir = os.path.join(split_dir, "images")
        labels_file = os.path.join(split_dir, "labels.txt")
        
        # 检查目录是否存在
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"图像目录不存在: {images_dir}")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"标签文件不存在: {labels_file}")
        
        # 加载标签
        with open(labels_file, 'r') as f:
            labels = [int(line.strip()) for line in f.readlines()]
        
        # 加载图像
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        n_samples = len(image_files)
        
        if n_samples != len(labels):
            raise ValueError(f"图像数量({n_samples})与标签数量({len(labels)})不匹配")
        
        print(f"  正在加载 {split} 集 ({n_samples} 张图片)...")
        
        # 预分配数组
        X = np.zeros((n_samples, 784), dtype=np.float32)
        
        # 逐张读取图像
        for i, img_path in enumerate(image_files):
            img = Image.open(img_path).convert('L')  # 转换为灰度图
            img_array = np.array(img, dtype=np.float32)
            
            # 归一化到 [0, 1]
            if self.normalize:
                img_array = img_array / 255.0
            
            # 展平为 784 维向量
            X[i] = img_array.flatten()
            
            # 显示进度
            if (i + 1) % 10000 == 0:
                print(f"    已加载 {i + 1}/{n_samples} 张图片")
        
        y = np.array(labels, dtype=np.int32)
        
        return X, y
    
    def reset_epoch(self):
        """重置epoch，重新打乱数据索引"""
        n_samples = self.X_train.shape[0]
        
        if self.shuffle:
            # 生成随机索引
            self.train_indices = np.random.permutation(n_samples)
        else:
            # 保持原始顺序
            self.train_indices = np.arange(n_samples)
        
        self.current_index = 0
        self.epoch_count += 1
        
        if self.epoch_count > 1:
            print(f"\n--- Epoch {self.epoch_count} 开始 ---")
    
    def get_next_batch(self):
        """
        获取下一个mini-batch
        
        返回:
            X_batch: 批次图像数据，形状为 (batch_size, 784)
            y_batch: 批次标签，形状为 (batch_size,)
            is_epoch_end: 是否为epoch的最后一个batch
        
        异常:
            StopIteration: 如果当前epoch的数据已全部返回
        """
        # 如果是第一次调用或已遍历完所有数据，重置epoch
        if self.train_indices is None or self.current_index >= len(self.train_indices):
            self.reset_epoch()
        
        n_samples = len(self.train_indices)
        start_idx = self.current_index
        end_idx = min(start_idx + self.batch_size, n_samples)
        
        # 获取当前batch的索引
        batch_indices = self.train_indices[start_idx:end_idx]
        
        # 提取数据
        X_batch = self.X_train[batch_indices]
        y_batch = self.y_train[batch_indices]
        
        # 更新当前位置
        self.current_index = end_idx
        
        # 判断是否为epoch结束
        is_epoch_end = (self.current_index >= n_samples)
        
        return X_batch, y_batch, is_epoch_end
    
    def get_all_batches(self):
        """
        获取当前epoch的所有mini-batch（生成器方式）
        
        Yields:
            X_batch: 批次图像数据
            y_batch: 批次标签
            batch_idx: 批次索引
            total_batches: 总批次数
        """
        self.reset_epoch()
        n_samples = len(self.train_indices)
        total_batches = int(np.ceil(n_samples / self.batch_size))
        
        for batch_idx in range(total_batches):
            X_batch, y_batch, _ = self.get_next_batch()
            yield X_batch, y_batch, batch_idx, total_batches
    
    def one_hot_encode(self, y, n_classes=10):
        """
        将标签转换为one-hot编码
        
        参数:
            y: 标签数组，形状为 (batch_size,) 或 (n_samples,)
            n_classes: 类别数量
        
        返回:
            y_onehot: one-hot编码后的标签，形状为 (n_classes, batch_size)
        """
        m = y.shape[0]
        y_onehot = np.zeros((n_classes, m), dtype=np.float32)
        y_onehot[y.astype(int), np.arange(m)] = 1.0
        return y_onehot
    
    def get_test_data(self, one_hot=False):
        """
        获取完整的测试集数据
        
        参数:
            one_hot: 是否将标签转换为one-hot编码
        
        返回:
            X_test: 测试集图像数据
            y_test: 测试集标签（原始或one-hot编码）
        """
        if one_hot:
            return self.X_test.T, self.one_hot_encode(self.y_test)
        else:
            return self.X_test.T, self.y_test
    
    def get_train_info(self):
        """获取训练集信息"""
        return {
            'total_samples': self.X_train.shape[0],
            'batch_size': self.batch_size,
            'total_batches_per_epoch': int(np.ceil(self.X_train.shape[0] / self.batch_size)),
            'current_epoch': self.epoch_count,
            'feature_dim': self.X_train.shape[1],
            'n_classes': len(np.unique(self.y_train))
        }


# ==================== 使用示例 ====================
def example_usage():
    """演示如何使用MNISTDataLoader"""
    
    print("=" * 60)
    print("MNIST Data Loader 使用示例")
    print("=" * 60)
    
    # 创建数据加载器
    loader = MNISTDataLoader(
        batch_size=128,
        shuffle=True,
        normalize=True
    )
    
    # 获取训练集信息
    info = loader.get_train_info()
    print(f"\n训练集信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 方法1: 手动获取每个batch
    print("\n--- 方法1: 手动获取batch ---")
    batch_count = 0
    while True:
        try:
            X_batch, y_batch, is_epoch_end = loader.get_next_batch()
            batch_count += 1
            print(f"Batch {batch_count}: X shape={X_batch.shape}, y shape={y_batch.shape}, epoch_end={is_epoch_end}")
            
            if is_epoch_end:
                print(f"✓ Epoch {loader.epoch_count} 完成，共 {batch_count} 个batch")
                break
        except StopIteration:
            break
    
    # 方法2: 使用生成器遍历所有batch
    print("\n--- 方法2: 使用生成器 ---")
    for X_batch, y_batch, batch_idx, total_batches in loader.get_all_batches():
        if batch_idx < 3:  # 只显示前3个batch
            print(f"Batch {batch_idx + 1}/{total_batches}: X shape={X_batch.shape}, y shape={y_batch.shape}")
    
    # 获取测试集
    print("\n--- 获取测试集 ---")
    X_test, y_test = loader.get_test_data(one_hot=False)
    print(f"测试集: X shape={X_test.shape}, y shape={y_test.shape}")
    
    # One-hot编码示例
    print("\n--- One-hot编码示例 ---")
    sample_labels = np.array([5, 0, 4, 1, 9])
    one_hot = loader.one_hot_encode(sample_labels)
    print(f"原始标签: {sample_labels}")
    print(f"One-hot形状: {one_hot.shape}")
    print(f"One-hot示例:\n{one_hot}")


if __name__ == "__main__":
    example_usage()
