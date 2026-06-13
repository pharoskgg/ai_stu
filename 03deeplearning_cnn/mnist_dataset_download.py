# 下载 MNIST 手写数字数据集
# 数据来源：MNIST 官方数据集（IDX 格式，直接下载）
# 数据集包含 70000 张 28x28 灰度手写数字图片，标签为 0-9
# 训练集 60000 张，测试集 10000 张

import numpy as np
import os
import time
import gzip
import struct
from urllib import request

# 设置保存路径
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(SAVE_DIR, "mnist_dataset.npz")

# MNIST 数据集下载地址（IDX 格式，来自官方镜像）
MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


class DownloadProgressBar:
    """urllib.urlretrieve 的进度条回调"""

    def __init__(self, desc="下载中"):
        self.desc = desc
        self.start_time = None

    def __call__(self, block_num, block_size, total_size):
        if self.start_time is None:
            self.start_time = time.time()
        downloaded = block_num * block_size
        if total_size <= 0:
            total_size = -1
        elapsed = max(time.time() - self.start_time, 1e-6)
        speed = downloaded / elapsed
        bar_width = 35

        if total_size > 0:
            ratio = min(downloaded / total_size, 1.0)
            filled = int(bar_width * ratio)
            bar = "█" * filled + "░" * (bar_width - filled)
            percent = ratio * 100
            size_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            if ratio < 1.0 and speed > 0:
                eta = (total_size - downloaded) / speed
                eta_str = f"ETA {eta:.0f}s"
            else:
                eta_str = ""
            print(
                f"\r  {self.desc} |{bar}| {percent:5.1f}% "
                f"{size_mb:.1f}/{total_mb:.1f}MB "
                f"速度 {speed / 1024:.0f}KB/s {eta_str}",
                end="", flush=True,
            )
        else:
            size_mb = downloaded / (1024 * 1024)
            print(
                f"\r  {self.desc} 已下载 {size_mb:.1f}MB "
                f"速度 {speed / 1024:.0f}KB/s",
                end="", flush=True,
            )
        if downloaded >= total_size > 0:
            print()


def _download_file(url, save_path, desc="下载中"):
    """下载文件并在本地保存，支持进度条显示"""
    if os.path.exists(save_path):
        print(f"  {desc}：使用本地缓存 {os.path.basename(save_path)}")
        return
    print(f"  {desc}：正在下载 {os.path.basename(save_path)}...")
    progress = DownloadProgressBar(desc=desc)
    request.urlretrieve(url, save_path, reporthook=progress)


def _parse_idx_labels(filepath):
    """解析 IDX 格式的标签文件（ubyte）"""
    with gzip.open(filepath, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"标签文件 magic number 错误: 期望 2049，实际 {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)


def _parse_idx_images(filepath):
    """解析 IDX 格式的图像文件（ubyte），返回 (N, 784) 的 float32 数组"""
    with gzip.open(filepath, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"图像文件 magic number 错误: 期望 2051，实际 {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(count, rows * cols).astype(np.float32)


total_start = time.time()

# ==================== 1. 下载数据集 ====================
step_start = time.time()
print("=" * 50)
print("[1/5] 正在下载 MNIST 数据集（首次下载可能需要几分钟）...")

# 下载 4 个 IDX 文件（带进度条），已缓存则跳过
local_files = {}
for key, filename in MNIST_FILES.items():
    url = MNIST_BASE_URL + filename
    save_path = os.path.join(SAVE_DIR, filename)
    _download_file(url, save_path, desc=f"[{key}]")
    local_files[key] = save_path

# 解析 IDX 文件为 numpy 数组
print("  正在解析数据集文件...")
y_train = _parse_idx_labels(local_files["train_labels"])
X_train = _parse_idx_images(local_files["train_images"])
y_test = _parse_idx_labels(local_files["test_labels"])
X_test = _parse_idx_images(local_files["test_images"])

# 拼接训练集和测试集，保持与原代码一致的数据形状
X = np.concatenate([X_train, X_test])   # (70000, 784) float32，像素值 0~255
y = np.concatenate([y_train, y_test])   # (70000,)     int32，标签 0~9
print(f"      ✔ 下载并解析完成，耗时 {time.time() - step_start:.1f}s")

step_start = time.time()
print("[2/5] 正在转换数据类型（float32 / int32）...")
# X 已经是 float32，y 已经是 int32，此处仅做归一化前的确认
print(f"      ✔ 类型确认完成，耗时 {time.time() - step_start:.1f}s")

# ==================== 2. 归一化 ====================
# 将像素值归一化到 [0, 1] 区间，便于神经网络训练
step_start = time.time()
print("[3/5] 正在归一化像素值到 [0, 1] 区间...")
X = X / 255.0
print(f"      ✔ 归一化完成，耗时 {time.time() - step_start:.1f}s")

# ==================== 3. 划分训练集和测试集 ====================
# MNIST 官方划分：前 60000 张为训练集，后 10000 张为测试集
step_start = time.time()
print("[4/5] 正在划分训练集（60000）和测试集（10000）...")
X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]
print(f"      ✔ 划分完成，耗时 {time.time() - step_start:.1f}s")

# ==================== 4. 保存为 npz 文件 ====================
step_start = time.time()
print("[5/5] 正在保存数据集到 npz 文件...")
np.savez(
    SAVE_PATH,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)
file_size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
print(f"      ✔ 保存完成，耗时 {time.time() - step_start:.1f}s，文件大小 {file_size_mb:.1f} MB")

print("=" * 50)
print(f"数据集已保存到: {SAVE_PATH}")
print(f"X_train shape: {X_train.shape}  (60000, 784)")
print(f"y_train shape: {y_train.shape}  (60000,)")
print(f"X_test  shape: {X_test.shape}  (10000, 784)")
print(f"y_test  shape: {y_test.shape}  (10000,)")
print(f"标签范围: {y_train.min()} ~ {y_train.max()}")
print(f"像素值范围: {X_train.min():.2f} ~ {X_train.max():.2f}")

# ==================== 5. 可视化示例 ====================
step_start = time.time()
print("正在生成可视化示例图片...")
import matplotlib.pyplot as plt
import platform

# 设置中文字体
system_name = platform.system()
if system_name == "Darwin":
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang HK"]
elif system_name == "Windows":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
else:
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(3, 5, figsize=(12, 8))
fig.suptitle("MNIST 手写数字数据集示例", fontsize=16)

for i, ax in enumerate(axes.flat):
    # 从训练集中随机取一张图
    idx = np.random.randint(0, len(X_train))
    # 将 784 维向量还原为 28x28 图片
    img = X_train[idx].reshape(28, 28)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"标签: {y_train[idx]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mnist_samples.png"), dpi=100)
plt.show()
print(f"✔ 示例图片已保存为 mnist_samples.png，耗时 {time.time() - step_start:.1f}s")
print("=" * 50)
print(f"全部完成！总耗时 {time.time() - total_start:.1f}s")
