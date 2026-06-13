"""
手势数字数据集加载器 - 支持Mini-Batch方式获取数据

功能说明:
1. 从data/hand_digital目录加载手势数字图片数据集
2. 自动将数据集划分为训练集和测试集（默认80%训练，20%测试）
3. 支持以mini-batch方式获取训练数据
4. 每轮(epoch)中确保不重复获取相同数据
5. 自动打乱数据顺序
6. 支持归一化和one-hot编码转换

目录结构:
data/hand_digital/
├── 0/  (约205张 .JPG 图片 - 数字0的手势)
├── 1/  (约206张 .JPG 图片 - 数字1的手势)
├── 2/  (约206张 .JPG 图片 - 数字2的手势)
├── ...
└── 9/  (约205张 .JPG 图片 - 数字9的手势)
"""

import numpy as np
import os
from PIL import Image
import glob


class HandDigitalDataLoader:
    """手势数字数据集加载器，支持mini-batch获取"""
    
    def __init__(self, data_dir=None, batch_size=64, shuffle=True, normalize=True, 
                 test_ratio=0.2, image_size=None, random_seed=42, filter_outliers=True, 
                 grayscale=True):
        """
        初始化数据加载器
        
        参数:
            data_dir: 数据集根目录路径（默认为当前脚本所在目录下的data/hand_digital文件夹）
            batch_size: mini-batch大小
            shuffle: 是否在每个epoch开始时打乱数据
            normalize: 是否将像素值归一化到[0, 1]区间
            test_ratio: 测试集比例（默认0.2，即20%）
            image_size: 图像resize的目标尺寸（宽, 高），默认为None（保持原始尺寸）。如果图片尺寸不一致，必须设置此参数
            random_seed: 随机种子，用于可复现的数据划分
            filter_outliers: 是否自动过滤异常尺寸的图片（默认True）。当检测到少数异常尺寸图片时，自动过滤掉它们
            grayscale: 是否将图片转换为灰度图（默认True）。如果为False，则保留RGB三通道
        """
        if data_dir is None:
            # 默认使用当前脚本所在目录下的data/hand_digital文件夹
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hand_digital")
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.test_ratio = test_ratio
        self.image_size = image_size
        self.random_seed = random_seed
        self.filter_outliers = filter_outliers
        self.grayscale = grayscale

        # 加载并划分数据集
        self.X_train, self.y_train, self.X_test, self.y_test = self._load_and_split_dataset()
        
        # 训练集相关状态
        self.train_indices = None  # 当前epoch的索引顺序
        self.current_index = 0  # 当前读取位置
        self.epoch_count = 0  # 已完成的epoch数
        
        print(f"✓ 训练集加载完成: {self.X_train.shape[0]} 个样本")
        print(f"✓ 测试集加载完成: {self.X_test.shape[0]} 个样本")
        print(f"✓ Batch size: {batch_size}, Shuffle: {shuffle}, Normalize: {normalize}, Grayscale: {grayscale}")
        if image_size is not None:
            print(f"✓ 图像尺寸: {image_size} (已resize), 特征维度: {image_size[0] * image_size[1]}")
        else:
            print(f"✓ 图像尺寸: 保持原始尺寸")

    def _load_and_split_dataset(self):
        """
        加载所有类别的数据并划分为训练集和测试集
        
        返回:
            X_train: 训练集图像数据，形状为 (n_train, height*width)
            y_train: 训练集标签，形状为 (n_train,)
            X_test: 测试集图像数据，形状为 (n_test, height*width)
            y_test: 测试集标签，形状为 (n_test,)
        """
        all_images = []
        all_labels = []
        image_sizes = []  # 记录所有图片的尺寸
        
        # 第一遍：收集所有图片的路径和尺寸
        image_paths_with_labels = []
        
        # 遍历0-9十个类别文件夹
        n_classes = 10
        for digit in range(n_classes):
            class_dir = os.path.join(self.data_dir, str(digit))
            
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"类别目录不存在: {class_dir}")
            
            # 获取该类别下所有图片文件（支持.JPG和.jpg）
            image_files = sorted(glob.glob(os.path.join(class_dir, "*.JPG")) + 
                                glob.glob(os.path.join(class_dir, "*.jpg")))
            
            if len(image_files) == 0:
                print(f"  ⚠ 警告: 类别 {digit} 目录下没有找到图片文件")
                continue
            
            print(f"  正在扫描类别 {digit}: {len(image_files)} 张图片")
            
            # 收集图片路径和标签
            for img_path in image_files:
                try:
                    with Image.open(img_path) as img:
                        # 根据grayscale参数决定是否转换为灰度图
                        if self.grayscale:
                            img = img.convert('L')  # 转换为灰度图
                        else:
                            img = img.convert('RGB')  # 保持RGB三通道
                        image_sizes.append(img.size)  # (width, height)
                        image_paths_with_labels.append((img_path, digit))
                except Exception as e:
                    print(f"  ⚠ 警告: 无法读取图片 {img_path}: {e}")
                    continue
        
        if len(image_paths_with_labels) == 0:
            raise ValueError("没有成功加载任何图片")
        
        # 输出所有图片的尺寸统计信息
        print(f"\n  📊 图片尺寸统计:")
        print(f"  总图片数: {len(image_sizes)}")
        
        # 统计不同尺寸的数量
        size_counter = {}
        for size in image_sizes:
            size_counter[size] = size_counter.get(size, 0) + 1
        
        # 按数量降序排列
        sorted_sizes = sorted(size_counter.items(), key=lambda x: x[1], reverse=True)
        
        print(f"  不同尺寸种类: {len(sorted_sizes)}")
        print(f"  尺寸分布 (宽x高: 数量):")
        for size, count in sorted_sizes[:20]:  # 只显示前20种最常见的尺寸
            percentage = (count / len(image_sizes)) * 100
            print(f"    {size[0]}x{size[1]}: {count} 张 ({percentage:.1f}%)")
        
        if len(sorted_sizes) > 20:
            print(f"    ... 还有 {len(sorted_sizes) - 20} 种其他尺寸")
        
        # 计算尺寸的统计信息
        widths = [s[0] for s in image_sizes]
        heights = [s[1] for s in image_sizes]
        print(f"\n  宽度统计:")
        print(f"    最小: {min(widths)}, 最大: {max(widths)}, 平均: {sum(widths)/len(widths):.1f}")
        print(f"  高度统计:")
        print(f"    最小: {min(heights)}, 最大: {max(heights)}, 平均: {sum(heights)/len(heights):.1f}")
        
        # 确定目标尺寸
        if self.image_size is None:
            # 未指定image_size，检查所有图片尺寸是否一致
            unique_sizes = set(image_sizes)
            if len(unique_sizes) == 1:
                # 所有图片尺寸一致，使用该尺寸
                target_size = list(unique_sizes)[0]
                print(f"\n  ✓ 检测到所有图片尺寸一致: {target_size}，将保持原始尺寸")
            else:
                # 图片尺寸不一致
                # 找出最常见的尺寸（主尺寸）
                most_common_size = sorted_sizes[0][0]
                most_common_count = sorted_sizes[0][1]
                outlier_count = len(image_sizes) - most_common_count
                outlier_percentage = (outlier_count / len(image_sizes)) * 100
                
                print(f"\n  ⚠ 检测到图片尺寸不一致:")
                print(f"     主要尺寸: {most_common_size[0]}x{most_common_size[1]} ({most_common_count}张, {100-outlier_percentage:.1f}%)")
                print(f"     异常尺寸: {outlier_count}张 ({outlier_percentage:.1f}%)")
                
                if self.filter_outliers and outlier_percentage < 5:
                    # 如果异常图片比例小于5%，自动过滤
                    print(f"  ✓ 自动过滤 {outlier_count} 张异常尺寸图片")
                    
                    # 过滤掉异常尺寸的图片
                    filtered_paths = []
                    for img_path, digit in image_paths_with_labels:
                        with Image.open(img_path) as img:
                            if img.size == most_common_size:
                                filtered_paths.append((img_path, digit))
                    
                    image_paths_with_labels = filtered_paths
                    target_size = most_common_size
                    print(f"  ✓ 过滤后剩余 {len(image_paths_with_labels)} 张图片，使用尺寸: {target_size}")
                else:
                    # 异常图片比例较高或用户禁用了过滤，抛出错误
                    min_width = min(s[0] for s in image_sizes)
                    max_width = max(s[0] for s in image_sizes)
                    min_height = min(s[1] for s in image_sizes)
                    max_height = max(s[1] for s in image_sizes)
                    raise ValueError(
                        f"❌ 检测到图片尺寸不一致！\n"
                        f"   尺寸范围: 宽[{min_width}-{max_width}], 高[{min_height}-{max_height}]\n"
                        f"   请设置 image_size 参数来统一resize所有图片，例如:\n"
                        f"   HandDigitalDataLoader(image_size=(28, 28))  # 或 (64, 64), (128, 128) 等\n"
                        f"   或者设置 filter_outliers=True 自动过滤少量异常图片"
                    )
        else:
            target_size = self.image_size
            print(f"\n  将图片统一resize到指定尺寸: {target_size}")
        
        # 第二遍：加载并处理所有图片
        print(f"  正在加载 {len(image_paths_with_labels)} 张图片...")
        for idx, (img_path, digit) in enumerate(image_paths_with_labels):
            try:
                # 根据grayscale参数决定转换模式
                if self.grayscale:
                    img = Image.open(img_path).convert('L')  # 转换为灰度图
                else:
                    img = Image.open(img_path).convert('RGB')  # 保持RGB三通道
                
                # 如果需要，resize到目标尺寸
                if img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                img_array = np.array(img, dtype=np.float32)
                
                # 归一化到 [0, 1]
                if self.normalize:
                    img_array = img_array / 255.0
                
                # 展平为一维向量
                all_images.append(img_array.flatten())
                all_labels.append(digit)
                
                # 显示进度
                if (idx + 1) % 500 == 0:
                    print(f"    已加载 {idx + 1}/{len(image_paths_with_labels)} 张图片")
            except Exception as e:
                print(f"  ⚠ 警告: 无法处理图片 {img_path}: {e}")
                continue
        
        if len(all_images) == 0:
            raise ValueError("没有成功加载任何图片")
        
        # 转换为numpy数组
        X = np.array(all_images, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        
        print(f"\n  总样本数: {len(X)}, 特征维度: {X.shape[1]}")
        
        # 按类别分层划分训练集和测试集
        X_train, y_train, X_test, y_test = self._stratified_split(X, y, self.test_ratio)
        
        return X_train, y_train, X_test, y_test
    
    def _stratified_split(self, X, y, test_ratio):
        """
        分层抽样划分训练集和测试集，确保每个类别在训练集和测试集中都有代表
        
        参数:
            X: 所有图像数据
            y: 所有标签
            test_ratio: 测试集比例
        
        返回:
            X_train, y_train, X_test, y_test
        """
        np.random.seed(self.random_seed)
        
        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []
        
        # 对每个类别分别划分
        for digit in range(10):
            # 获取当前类别的索引
            class_indices = np.where(y == digit)[0]
            n_samples = len(class_indices)
            
            if n_samples == 0:
                continue
            
            # 打乱索引
            np.random.shuffle(class_indices)
            
            # 计算测试集数量
            n_test = int(n_samples * test_ratio)
            
            # 划分
            test_indices = class_indices[:n_test]
            train_indices = class_indices[n_test:]
            
            X_test_list.append(X[test_indices])
            y_test_list.append(y[test_indices])
            X_train_list.append(X[train_indices])
            y_train_list.append(y[train_indices])
        
        # 合并所有类别的数据
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)
        
        # 再次打乱训练集和测试集
        train_shuffle_idx = np.random.permutation(len(X_train))
        test_shuffle_idx = np.random.permutation(len(X_test))
        
        X_train = X_train[train_shuffle_idx]
        y_train = y_train[train_shuffle_idx]
        X_test = X_test[test_shuffle_idx]
        y_test = y_test[test_shuffle_idx]
        
        print(f"  训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
        print(f"  训练集类别分布: {np.bincount(y_train.astype(int))}")
        print(f"  测试集类别分布: {np.bincount(y_test.astype(int))}")
        
        return X_train, y_train, X_test, y_test
    
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
            X_batch: 批次图像数据，形状为 (batch_size, height*width)
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
            X_test: 测试集图像数据，形状为 (n_features, n_samples)
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
            'test_samples': self.X_test.shape[0],
            'batch_size': self.batch_size,
            'total_batches_per_epoch': int(np.ceil(self.X_train.shape[0] / self.batch_size)),
            'current_epoch': self.epoch_count,
            'feature_dim': self.X_train.shape[1],
            'image_size': self.image_size if self.image_size is not None else '原始尺寸',
            'n_classes': len(np.unique(self.y_train)),
            'test_ratio': self.test_ratio,
            'grayscale': self.grayscale
        }


# ==================== 使用示例 ====================
def example_usage():
    """演示如何使用HandDigitalDataLoader"""
    
    print("=" * 60)
    print("手势数字数据加载器使用示例")
    print("=" * 60)
    
    # 示例1: 指定image_size进行resize，使用灰度图（默认）
    print("\n【示例1】指定image_size=(28, 28)进行resize，灰度图模式:")
    print("-" * 60)
    loader = HandDigitalDataLoader(
        batch_size=128,
        shuffle=True,
        normalize=True,
        test_ratio=0.2,
        image_size=(28, 28),  # 明确设置目标尺寸
        grayscale=True  # 使用灰度图（默认值，可省略）
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
