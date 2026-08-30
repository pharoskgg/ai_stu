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
learn_rate = 0.4  # 学习率
loop_count = 20000
dropout_rate = 0.2  # Dropout率

model = KSNet.KSSequential(
    KSNet.KSLinear(input_dim=2, output_dim=h),
    KSNet.KSReLU(),
    KSNet.KSDropout(dropout_rate=dropout_rate),
    KSNet.KSLinear(h, 1)
)

logit_loss = KSNet.KSBinaryLogisticLoss()
optimizer = KSNet.KSSGDOptimizer(model.parameters(), lr=learn_rate)

for epoch in range(loop_count):

    loss = model.train_step(
        x=X_train,
        label=Y_train,
        loss_fn=logit_loss,
        optimizer=optimizer,
    )

    if epoch % 1000 == 0:
        losses.append(loss)

        print(f"Iteration {epoch}, Loss: {loss:.6f}")


    


    




