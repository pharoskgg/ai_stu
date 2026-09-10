import sys
from pathlib import Path

# 把项目根目录 ai_stu 加进 sys.path，让 import KSNet 能找到包
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import KSNet
import numpy as np
from KSNet.util import (generate_number_seq2seq_dataset,get_seq2seq_data,)

dataset = generate_number_seq2seq_dataset(n_samples=10000, random_state=42,shared_vocab=True,)

train_arb2zh = get_seq2seq_data(dataset, direction="arabic_to_chinese",split="train",)

train_zh2arb = get_seq2seq_data(dataset,direction="chinese_to_arabic",split="train",)

source_ids = train_arb2zh["source_ids"]
decoder_input_ids = train_arb2zh["decoder_input_ids"]
decoder_target_ids = train_arb2zh["decoder_target_ids"]

source_vocab = train_arb2zh["source_vocab"]
target_vocab = train_arb2zh["target_vocab"]

print(source_ids.shape)
print(decoder_input_ids.shape)
print(decoder_target_ids.shape)
print(len(source_vocab))  # 28
print(len(target_vocab))  # 28

"""
embed_dim = 8
encoder_hidden_dim = 16
vocab_size = dataset.vocab_size

traindata = None

tearch_force = None
label_data = None

encoder_embedding = KSNet.KSEmbedding(vocab_size=vocab_size, embedding_dim=embed_dim)
encoder_rnn = KSNet.KSRnn(input_size=embed_dim, hidden_size=encoder_hidden_dim)

decoder_embedding = KSNet.KSEmbedding(vocab_size=vocab_size, embedding_dim=embed_dim)
decoder_rnn = KSNet.KSRnn(input_size=embed_dim, hidden_size=encoder_hidden_dim)
decoder_output = KSNet.KSLinear(input_size=encoder_hidden_dim, output_size=vocab_size)

loss = KSNet.KSSoftmaxCrossEntropyLoss()


for x in traindata:
    encoder_input = encoder_embedding.forward(x)
    encoder_output = encoder_rnn.forward(encoder_input)

    decoder_input = decoder_embedding.forward(tearch_force)
    decoder_output, decoder_hidden = decoder_rnn.forward(decoder_input, encoder_output)
    output = decoder_output.forward(decoder_output)

    loss_value = loss.forward(output, label_data)

"""