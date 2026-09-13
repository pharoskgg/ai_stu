"""数字 Seq2Seq 的数据准备示例。

这里只准备模型需要的三组 token id：

* ``encoder_input_ids``：编码器输入；
* ``decoder_input_ids``：以 ``<SOS>`` 开头的 teacher-forcing 输入；
* ``decoder_target_ids``：以 ``<EOS>`` 结尾的 decoder 标签。

同一份数字数据可以通过 ``direction`` 切换为两个翻译方向。
"""

import sys
from pathlib import Path

import numpy as np

# 把项目根目录 ai_stu 加进 sys.path，让 import KSNet 能找到包。
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from KSNet.util import generate_number_seq2seq_dataset, get_seq2seq_data
import KSNet

dataset = generate_number_seq2seq_dataset(n_samples=10_000, min_value=0, max_value=100_000)
# 阿拉伯数字 -> 中文数字
train_arabic_to_chinese = get_seq2seq_data(dataset, direction="arabic_to_chinese", split="train",)
# 中文数字 -> 阿拉伯数字
train_chinese_to_arabic = get_seq2seq_data(dataset, direction="chinese_to_arabic", split="train",)

# 方向一：阿拉伯数字 -> 中文数字
encoder_input_ids = train_arabic_to_chinese["encoder_input_ids"]
decoder_input_ids = train_arabic_to_chinese["decoder_input_ids"]
decoder_target_ids = train_arabic_to_chinese["decoder_target_ids"]

# 方向二：中文数字 -> 阿拉伯数字
reverse_encoder_input_ids = train_chinese_to_arabic["encoder_input_ids"]
reverse_decoder_input_ids = train_chinese_to_arabic["decoder_input_ids"]
reverse_decoder_target_ids = train_chinese_to_arabic["decoder_target_ids"]


def check_ready(name: str, data: dict) -> None:
    """检查一组 Seq2Seq 数据是否能直接喂给 encoder/decoder。"""
    encoder_ids = data["encoder_input_ids"]
    decoder_ids = data["decoder_input_ids"]
    labels = data["decoder_target_ids"]

    assert encoder_ids.ndim == 2
    assert decoder_ids.ndim == 2
    assert labels.ndim == 2
    assert decoder_ids.shape == labels.shape
    assert encoder_ids.shape[0] == decoder_ids.shape[0]
    assert np.all(
        data["decoder_target_mask"] == (labels != data["target_vocab"].pad_id)
    )

    sample = 0
    source_text = data["source_vocab"].decode(encoder_ids[sample])
    decoder_input_text = data["target_vocab"].decode(decoder_ids[sample])
    label_text = data["target_vocab"].decode(labels[sample])
    print(
        f"{name}: encoder={encoder_ids.shape}, "
        f"decoder_input={decoder_ids.shape}, label={labels.shape}"
    )
    print(
        f"  sample: {source_text} -> "
        f"decoder_input={decoder_input_text}, label={label_text}"
    )


# check_ready("arabic_to_chinese", train_arabic_to_chinese)
# check_ready("chinese_to_arabic", train_chinese_to_arabic)


embed_dim = dataset.vocab_size + 8
encoder_hidden_dim = dataset.vocab_size + 16
vocab_size = dataset.vocab_size

encoder_embedding = KSNet.KSEmbedding(vocab_size=vocab_size, embedding_dim=embed_dim)
encoder_rnn = KSNet.KSRnn(input_size=embed_dim, hidden_size=encoder_hidden_dim)

decoder_embedding = KSNet.KSEmbedding(vocab_size=vocab_size, embedding_dim=embed_dim)
decoder_rnn = KSNet.KSRnn(input_size=embed_dim, hidden_size=encoder_hidden_dim, return_sequences=True)
decoder_fnn = KSNet.KSLinear(input_dim=encoder_hidden_dim, output_dim=vocab_size)

loss = KSNet.KSSoftmaxCrossEntropyLoss()

trainable_layers = [encoder_embedding, encoder_rnn, decoder_embedding, decoder_rnn, decoder_fnn,]
params = [ param_and_grad
    for layer in trainable_layers
    for param_and_grad in layer.parameters()
]
optimizer = KSNet.KSAdamOptimizer(params, lr=0.001)

epochs = 100
for epoch in range(epochs):
    for i, seq in enumerate(encoder_input_ids):
        optimizer.zero_grad()
        # 提取有效长度，去除pad
        source_len = int(train_arabic_to_chinese["source_lengths"][i])
        decoder_len = int(train_arabic_to_chinese["decoder_lengths"][i])

        seq = seq[:source_len]
        target_input_ids = decoder_input_ids[i, :decoder_len]
        target_idx = decoder_target_ids[i, :decoder_len]

        # 开始训练
        encoder_input = encoder_embedding.forward(seq)
        encoder_hidden = encoder_rnn.forward(encoder_input)

        decoder_token = decoder_embedding.forward(target_input_ids)
        decoder_hiddens = decoder_rnn.forward(decoder_token, encoder_hidden)

        # decoder_hiddens = decoder_hiddens.reshape(-1, encoder_hidden_dim)
        decoder_outputs = decoder_fnn.forward(decoder_hiddens)

        loss_value = loss.forward(decoder_outputs, target_idx)

        print(loss_value)
        d_loss = loss.backward()
        d_decoder_hidden = decoder_fnn.backward(d_loss)
        d_decoder_input = decoder_rnn.backward(d_decoder_hidden)
        decoder_embedding.backward(d_decoder_input)

        d_encoder_input = encoder_rnn.backward(decoder_rnn.dh_initial)
        encoder_embedding.backward(d_encoder_input)
        optimizer.step()
