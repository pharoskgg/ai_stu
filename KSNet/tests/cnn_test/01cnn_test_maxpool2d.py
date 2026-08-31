import sys
from pathlib import Path

import numpy as np

# 支持从当前测试目录直接执行本文件。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KSNet.layers.KSMaxPool2D import KSMaxPool2D

x = np.array(
    [[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 1, 2, 3],
        [4, 5, 6, 7],
    ]]],
    dtype=np.float32,
)

pool = KSMaxPool2D(kernel_size=2, stride=2)

output = pool.forward(x)
print("output:")
print(output)

dout = np.ones_like(output)
dx = pool.backward(dout)

print("dx:")
print(dx)
