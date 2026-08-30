import sys
from pathlib import Path

# 把项目根目录 ai_stu 加进 sys.path，让 import KSNet 能找到包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import KSNet
import numpy as np
from KSNet.util import generate_moons_dataset

(X_train, Y_train), (X_test, Y_test) = generate_moons_dataset(n_samples=600)
# 转置为 (特征数, 样本数)，方便向量化计算
Y_train = Y_train.reshape(-1, 1)  # (240, 1)
Y_test = Y_test.reshape(-1, 1)  # (60, 1)

np.random.seed(42)
h = 8
losses = []
_lambda = 0.001  # L2正则化强度
learn_rate = 0.08  # 学习率
loop_count = 40000


linear1 = KSNet.KSLinear(input_dim=2, output_dim=h)
linear1_relu = KSNet.KSReLU()
linear2 = KSNet.KSLinear(h, 1)
logit_loss = KSNet.KSBinaryLogisticLoss()
layer_list = [linear1, linear1_relu, linear2]
params = [param for layer in layer_list for param in layer.parameters()]
optimizer = KSNet.KSSGDOptimizer(params=params, lr=learn_rate, weight_decay=_lambda)



for epoch in range(loop_count):
    a1 = linear1.forward(X_train)
    relu_a1 = linear1_relu.forward(a1)
    a2 = linear2.forward(relu_a1)

    loss = logit_loss.forward(a2, Y_train)
    # 与优化器中的 weight_decay 对应：对传入优化器的全部参数施加 L2。
    loss += _lambda / 2 * sum(np.sum(param ** 2) for param, _ in params)
    
    dout = logit_loss.backward()
    
    linear2_dout = linear2.backward(dout)
    linear1_relu_dout = linear1_relu.backward(linear2_dout)
    linear1_dout = linear1.backward(linear1_relu_dout)

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1000 == 0:
        losses.append(loss)

        print(f"Iteration {epoch}, Loss: {loss:.6f}")


    


    




