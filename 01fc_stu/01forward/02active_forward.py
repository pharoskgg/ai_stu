# 全连接层计算
import numpy as np

A = np.random.randn(1, 256)

W1 = np.random.randn(256, 64)
b1 = np.random.randn(64)

W2 = np.random.randn(64, 32)
b2 = np.random.randn(32)

W3 = np.random.randn(32, 10)
b3 = np.random.randn(10)

def fc_layer_forward(x, w, b):
    return np.dot(x, w) + b

print(A.shape)

out1 = fc_layer_forward(A, W1, b1) # => 1 * 64

out2 = fc_layer_forward(out1, W2, b2) # => 1 * 32

out3 = fc_layer_forward(out2, W3, b3) # => 1 * 10

print(out3)
