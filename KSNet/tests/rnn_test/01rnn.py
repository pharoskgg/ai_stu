"""使用 batch 训练双向数字 Seq2Seq，并在测试集上进行自回归评估。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import KSNet
from KSNet.util import generate_number_seq2seq_dataset, get_seq2seq_data


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
    print(
        f"{name}: encoder={encoder_ids.shape}, "
        f"decoder_input={decoder_ids.shape}, label={labels.shape}"
    )


def main(n_samples: int = 10_000, epochs: int = 300, batch_size: int = 32) -> None:
    dataset = generate_number_seq2seq_dataset(
        n_samples=n_samples,
        min_value=0,
        max_value=100_000,
        shared_vocab=True,
    )
    train_sets = {
        "ar2zh": get_seq2seq_data(dataset, "arabic_to_chinese", "train"),
        "zh2ar": get_seq2seq_data(dataset, "chinese_to_arabic", "train"),
    }
    test_sets = {
        "ar2zh": get_seq2seq_data(dataset, "arabic_to_chinese", "test"),
        "zh2ar": get_seq2seq_data(dataset, "chinese_to_arabic", "test"),
    }

    embed_dim = dataset.vocab_size + 8
    hidden_dim = dataset.vocab_size + 16
    vocab_size = dataset.vocab_size

    encoder_embedding = KSNet.KSEmbedding(vocab_size, embed_dim)
    encoder_rnn = KSNet.KSRnn(embed_dim, hidden_dim)
    decoder_embedding = KSNet.KSEmbedding(vocab_size, embed_dim)
    decoder_rnn = KSNet.KSRnn(embed_dim, hidden_dim, return_sequences=True)
    decoder_fnn = KSNet.KSLinear(hidden_dim, vocab_size, weight_init="xavier")
    loss = KSNet.KSSoftmaxCrossEntropyLoss()

    layers = [
        encoder_embedding,
        encoder_rnn,
        decoder_embedding,
        decoder_rnn,
        decoder_fnn,
    ]
    optimizer = KSNet.KSAdamOptimizer(
        [parameter for layer in layers for parameter in layer.parameters()],
        lr=0.001,
    )
    rng = np.random.default_rng(42)

    def train_batch(data: dict, batch_indices: np.ndarray):
        optimizer.zero_grad()
        source_max_len = int(np.max(data["source_lengths"][batch_indices]))
        decoder_max_len = int(np.max(data["decoder_lengths"][batch_indices]))

        source_ids = data["encoder_input_ids"][batch_indices, :source_max_len]
        source_mask = data["encoder_mask"][batch_indices, :source_max_len]
        decoder_ids = data["decoder_input_ids"][batch_indices, :decoder_max_len]
        targets = data["decoder_target_ids"][batch_indices, :decoder_max_len]
        target_mask = data["decoder_target_mask"][batch_indices, :decoder_max_len]

        encoder_hidden = encoder_rnn.forward(
            encoder_embedding.forward(source_ids), mask=source_mask
        )
        decoder_hiddens = decoder_rnn.forward(
            decoder_embedding.forward(decoder_ids), encoder_hidden, mask=target_mask
        )

        flat_outputs = decoder_fnn.forward(decoder_hiddens.reshape(-1, hidden_dim))
        flat_targets = targets.reshape(-1)
        flat_mask = target_mask.reshape(-1)
        loss_value = loss.forward(flat_outputs[flat_mask], flat_targets[flat_mask])

        # loss 只返回有效 token 的梯度，这里将它散射回完整的 (B*T, V)。
        d_outputs = np.zeros_like(flat_outputs)
        d_outputs[flat_mask] = loss.backward()
        d_decoder_hidden = decoder_fnn.backward(d_outputs).reshape(
            len(batch_indices), decoder_max_len, hidden_dim
        )
        decoder_embedding.backward(decoder_rnn.backward(d_decoder_hidden))
        encoder_embedding.backward(encoder_rnn.backward(decoder_rnn.dh_initial))
        optimizer.step()
        return loss_value, int(np.sum(flat_mask))

    def evaluate(data: dict):
        """使用 greedy decoding 评估，不给 decoder 喂真实答案。"""
        token_correct = 0
        token_total = 0
        sequence_correct = 0
        sequence_total = len(data["encoder_input_ids"])
        target_vocab = data["target_vocab"]
        max_decode_len = data["decoder_target_ids"].shape[1]

        for start in range(0, sequence_total, batch_size):
            batch_indices = np.arange(start, min(start + batch_size, sequence_total))
            source_max_len = int(np.max(data["source_lengths"][batch_indices]))
            source_ids = data["encoder_input_ids"][batch_indices, :source_max_len]
            source_mask = data["encoder_mask"][batch_indices, :source_max_len]
            targets = data["decoder_target_ids"][batch_indices]
            target_mask = data["decoder_target_mask"][batch_indices]

            hidden = encoder_rnn.forward(
                encoder_embedding.forward(source_ids), mask=source_mask
            )
            current_ids = np.full(
                len(batch_indices), target_vocab.sos_id, dtype=np.int64
            )
            predictions = np.full_like(targets, target_vocab.pad_id)
            active = np.ones(len(batch_indices), dtype=bool)

            for t in range(max_decode_len):
                step_hidden = decoder_rnn.forward(
                    decoder_embedding.forward(current_ids[:, None]),
                    hidden,
                    mask=active[:, None],
                )
                hidden = step_hidden[:, 0, :]
                next_ids = np.argmax(decoder_fnn.forward(hidden), axis=1)
                predictions[:, t] = np.where(active, next_ids, target_vocab.pad_id)
                active &= next_ids != target_vocab.eos_id
                current_ids = np.where(active, next_ids, target_vocab.pad_id)
                if not np.any(active):
                    break

            token_correct += int(np.sum((predictions == targets) & target_mask))
            token_total += int(np.sum(target_mask))
            sequence_correct += int(np.sum(np.all(predictions == targets, axis=1)))

        return token_correct / token_total, sequence_correct / sequence_total

    def predict(text: str, direction: str) -> str:
        """对一条用户输入执行 greedy 自回归解码。"""
        if direction == "ar2zh":
            source_ids = dataset.text_to_token_sequence(
                text,
                token_type="arabic",
                add_special_tokens=True,
                as_target=False,
            )
            source_vocab = train_sets[direction]["source_vocab"]
            target_vocab = train_sets[direction]["target_vocab"]
        elif direction == "zh2ar":
            source_ids = dataset.text_to_token_sequence(
                text,
                token_type="chinese",
                add_special_tokens=True,
                as_target=False,
            )
            source_vocab = train_sets[direction]["source_vocab"]
            target_vocab = train_sets[direction]["target_vocab"]
        else:
            raise ValueError("direction 必须是 ar2zh 或 zh2ar")

        source_ids = np.asarray(source_ids, dtype=np.int64)[None, :]
        source_mask = np.ones(source_ids.shape, dtype=bool)
        hidden = encoder_rnn.forward(
            encoder_embedding.forward(source_ids),
            mask=source_mask,
        )

        current_id = target_vocab.sos_id
        generated_ids = []
        # 数据范围为 0~100000，64 步足以覆盖两种方向的输出长度。
        for _ in range(64):
            step_hidden = decoder_rnn.forward(
                decoder_embedding.forward(
                    np.asarray([[current_id]], dtype=np.int64)
                ),
                hidden,
                mask=np.ones((1, 1), dtype=bool),
            )
            hidden = step_hidden[:, 0, :]
            logits = decoder_fnn.forward(hidden)
            current_id = int(np.argmax(logits[0]))
            if current_id == target_vocab.eos_id:
                break
            generated_ids.append(current_id)

        # source_vocab 用于明确记录当前方向的词表；实际输出使用 target_vocab。
        _ = source_vocab
        return target_vocab.decode(generated_ids)

    for epoch in range(epochs):
        # 混合两个方向的 mini-batch，避免模型连续偏向某一个翻译方向。
        jobs = []
        for direction, data in train_sets.items():
            indices = rng.permutation(len(data["encoder_input_ids"]))
            for start in range(0, len(indices), batch_size):
                jobs.append((direction, data, indices[start : start + batch_size]))
        rng.shuffle(jobs)

        loss_sums = {direction: 0.0 for direction in train_sets}
        token_counts = {direction: 0 for direction in train_sets}
        for direction, data, batch_indices in jobs:
            loss_value, token_count = train_batch(data, batch_indices)
            loss_sums[direction] += loss_value * token_count
            token_counts[direction] += token_count

        ar2zh_token_acc, ar2zh_exact_acc = evaluate(test_sets["ar2zh"])
        zh2ar_token_acc, zh2ar_exact_acc = evaluate(test_sets["zh2ar"])
        print(
            f"epoch={epoch + 1:4d} | "
            f"ar2zh loss={loss_sums['ar2zh'] / token_counts['ar2zh']:.6f}, "
            f"token_acc={ar2zh_token_acc:.2%}, exact_acc={ar2zh_exact_acc:.2%} | "
            f"zh2ar loss={loss_sums['zh2ar'] / token_counts['zh2ar']:.6f}, "
            f"token_acc={zh2ar_token_acc:.2%}, exact_acc={zh2ar_exact_acc:.2%}"
        )

    print("\n训练完成。现在可以输入数据进行测试（输入 q 退出）。")
    print("方向 1：阿拉伯数字 -> 中文数字，例如 1234")
    print("方向 2：中文数字 -> 阿拉伯数字，例如 一千二百三十四")
    while True:
        direction_input = input("请选择方向 [1/2/q]: ").strip().lower()
        if direction_input in {"q", "quit", "exit"}:
            break
        direction = {"1": "ar2zh", "2": "zh2ar"}.get(direction_input)
        if direction is None:
            print("请输入 1、2 或 q。")
            continue

        text = input("请输入数据: ").strip()
        if not text:
            print("输入不能为空。")
            continue
        try:
            result = predict(text, direction)
            print(f"预测结果: {result}")
        except (TypeError, ValueError) as exc:
            print(f"输入无效: {exc}")


if __name__ == "__main__":
    main()
