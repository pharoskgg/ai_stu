from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split


# 这些 token 的顺序固定，便于在不同实验之间复用词表。
SPECIAL_TOKENS = ("<PAD>", "<SOS>", "<EOS>", "<UNK>")
CHINESE_NUMBER_DIGITS = "零一二三四五六七八九"
CHINESE_NUMBER_UNITS = "十百千万"
MAX_CHINESE_NUMBER = 100_000


@dataclass
class Vocabulary:
    """一个适用于字符级 Seq2Seq 的简单词表。

    ``token_to_id`` 和 ``id_to_token`` 都是公开属性，训练代码可以直接使用。
    默认保留 ``<PAD>``、``<SOS>``、``<EOS>`` 和 ``<UNK>`` 四个特殊 token。
    """

    token_to_id: Mapping[str, int]
    id_to_token: Sequence[str]

    def __post_init__(self):
        # 拷贝为普通容器，避免调用者修改传入列表后词表悄悄发生变化。
        self.token_to_id = dict(self.token_to_id)
        self.id_to_token = list(self.id_to_token)

    @classmethod
    def from_tokens(
        cls,
        tokens: Iterable[str],
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> "Vocabulary":
        """根据 token 序列建立词表，并去除重复 token。"""
        ordered_tokens: List[str] = []
        for token in list(special_tokens) + list(tokens):
            if token not in ordered_tokens:
                ordered_tokens.append(token)
        return cls(
            token_to_id={token: index for index, token in enumerate(ordered_tokens)},
            id_to_token=ordered_tokens,
        )

    def __len__(self) -> int:
        return len(self.id_to_token)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def __getitem__(self, token: str) -> int:
        """按 token 取 id；不存在的 token 映射到 ``<UNK>``。"""
        return self.token_to_id.get(token, self.unk_id)

    @property
    def stoi(self) -> Mapping[str, int]:
        """``string-to-index`` 别名，兼容常见 Seq2Seq 写法。"""
        return self.token_to_id

    @property
    def itos(self) -> Sequence[str]:
        """``index-to-string`` 别名，兼容常见 Seq2Seq 写法。"""
        return self.id_to_token

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def sos_id(self) -> int:
        return self.token_to_id["<SOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    def encode(
        self,
        text: Union[str, Sequence[str]],
        add_sos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """将字符串（或 token 序列）编码为 token id 序列。"""
        tokens = list(text) if isinstance(text, str) else list(text)
        encoded: List[int] = []
        if add_sos:
            encoded.append(self.sos_id)
        encoded.extend(self.token_to_id.get(token, self.unk_id) for token in tokens)
        if add_eos:
            encoded.append(self.eos_id)
        return encoded

    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """将 token id 序列还原为字符串。"""
        special_tokens = set(SPECIAL_TOKENS)
        tokens = []
        for index in ids:
            index = int(index)
            if index < 0 or index >= len(self.id_to_token):
                token = "<UNK>"
            else:
                token = self.id_to_token[index]
            if skip_special_tokens and token in special_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)

    def to_dict(self) -> dict:
        """返回便于保存到 JSON 的词表字典。"""
        return dict(self.token_to_id)


def _pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_id: int,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """将不等长 id 序列右侧补齐为 ``(batch, length)`` 的数组。"""
    if not sequences:
        length = 0 if max_length is None else max_length
        return np.empty((0, length), dtype=np.int64)
    if max_length is None:
        max_length = max(len(sequence) for sequence in sequences)
    if max_length < 0:
        raise ValueError("max_length 不能小于 0")
    if any(len(sequence) > max_length for sequence in sequences):
        raise ValueError("存在长度超过 max_length 的序列")

    result = np.full((len(sequences), max_length), pad_id, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        result[row, : len(sequence)] = sequence
    return result


def number_to_chinese(number: int) -> str:
    """将 ``0`` 到 ``100000`` 的非负整数转换为中文数字。

    例如：

    .. code-block:: python

        number_to_chinese(0)       # "零"
        number_to_chinese(10101)   # "一万零一百零一"
        number_to_chinese(99900)   # "九万九千九百"
        number_to_chinese(100000)  # "十万"

    这里使用字符级中文数字，适合直接作为 Seq2Seq 的目标序列。
    """
    if isinstance(number, bool) or not isinstance(number, (int, np.integer)):
        raise TypeError("number 必须是整数")
    number = int(number)
    if number < 0 or number > MAX_CHINESE_NUMBER:
        raise ValueError(f"number 必须位于 [0, {MAX_CHINESE_NUMBER}] 范围内")
    if number == 0:
        return "零"

    def section_to_chinese(section: int, omit_one_before_ten: bool = False) -> str:
        """转换小于一万的一段数字。"""
        result: List[str] = []
        zero_pending = False
        for position in range(3, -1, -1):
            divisor = 10**position
            digit = (section // divisor) % 10
            if digit:
                if zero_pending:
                    result.append("零")
                # 只有整段数字以“十”开头时省略“一”；例如 10 -> 十，
                # 但 1010 -> 一千零一十，不能写成“一千零十”。
                if not (
                    omit_one_before_ten
                    and position == 1
                    and digit == 1
                    and not result
                ):
                    result.append(CHINESE_NUMBER_DIGITS[digit])
                result.append(("", "十", "百", "千")[position])
                zero_pending = False
            elif result and section % divisor != 0:
                # 中间有断位且后面还有非零数字时，下一次输出前补一个零。
                zero_pending = True
        return "".join(result)

    if number < 10_000:
        return section_to_chinese(number, omit_one_before_ten=True)

    ten_thousands, remainder = divmod(number, 10_000)
    result = section_to_chinese(ten_thousands, omit_one_before_ten=True) + "万"
    if remainder:
        if remainder < 1_000:
            result += "零"
        # “万”后面的分段不是整句开头，10 应写作“一十”，例如
        # 10010 -> 一万零一十。
        result += section_to_chinese(remainder, omit_one_before_ten=False)
    return result


def build_number_vocab(
    shared: bool = True,
) -> Tuple[Vocabulary, Vocabulary]:
    """建立数字 Seq2Seq 词表。

    默认返回同一个共享词表两次，即 ``source_vocab is target_vocab``。共享词表
    包含阿拉伯数字、中文数字、中文单位和四个特殊 token，适合双向翻译以及
    共享 embedding。传入 ``shared=False`` 可以恢复为两个独立词表。
    """
    if not isinstance(shared, bool):
        raise TypeError("shared 必须是 bool")

    if shared:
        shared_vocab = Vocabulary.from_tokens(
            "0123456789" + CHINESE_NUMBER_DIGITS + CHINESE_NUMBER_UNITS
        )
        return shared_vocab, shared_vocab

    source_vocab = Vocabulary.from_tokens("0123456789")
    target_vocab = Vocabulary.from_tokens(
        CHINESE_NUMBER_DIGITS + CHINESE_NUMBER_UNITS
    )
    return source_vocab, target_vocab


def build_shared_vocab() -> Vocabulary:
    """建立一份同时包含阿拉伯和中文 token 的共享词表。"""
    return build_number_vocab(shared=True)[0]


def build_vocab(shared: bool = True) -> Tuple[Vocabulary, Vocabulary]:
    """``build_number_vocab`` 的简短别名，默认建立共享词表。"""
    return build_number_vocab(shared=shared)


@dataclass
class Seq2SeqSplit:
    """Seq2Seq 数据集的一个划分（训练集或测试集）。"""

    numbers: np.ndarray
    source_text: List[str]
    target_text: List[str]
    source_ids: np.ndarray
    target_ids: np.ndarray
    decoder_input_ids: np.ndarray
    decoder_target_ids: np.ndarray
    source_lengths: np.ndarray
    target_lengths: np.ndarray

    def __len__(self) -> int:
        return len(self.numbers)

    @property
    def encoder_input_ids(self) -> np.ndarray:
        """编码器输入 token ids 的直观别名。

        ``source_ids`` 保留用于兼容已有代码；在 Seq2Seq 训练代码中，
        ``encoder_input_ids`` 更明确地表达了这组序列的用途。
        """
        return self.source_ids

    @property
    def decoder_lengths(self) -> np.ndarray:
        """decoder 每个样本的有效时间步数量（不包含最后的 padding）。"""
        # target_ids 是 <SOS> + target + <EOS>，而 decoder 的输入/标签
        # 分别是右移后的两半，因此有效长度比 target_lengths 少 1。
        return self.target_lengths - 1

    @property
    def encoder_mask(self) -> np.ndarray:
        """编码器 padding mask；有效 token 为 True。"""
        return self.source_ids != 0

    @property
    def decoder_target_mask(self) -> np.ndarray:
        """decoder 标签 padding mask；有效标签为 True。"""
        return self.decoder_target_ids != 0

    @property
    def decoder_input_mask(self) -> np.ndarray:
        """decoder 输入 padding mask；有效 token 为 True。"""
        return self.decoder_input_ids != 0

    def __getitem__(self, index: int) -> dict:
        """按样本取出训练 Seq2Seq 所需字段。"""
        return {
            "number": int(self.numbers[index]),
            "source_text": self.source_text[index],
            "target_text": self.target_text[index],
            "source_ids": self.source_ids[index],
            "encoder_input_ids": self.encoder_input_ids[index],
            "target_ids": self.target_ids[index],
            "decoder_input_ids": self.decoder_input_ids[index],
            "decoder_target_ids": self.decoder_target_ids[index],
            "source_length": int(self.source_lengths[index]),
            "target_length": int(self.target_lengths[index]),
            "decoder_length": int(self.decoder_lengths[index]),
            "encoder_mask": self.encoder_mask[index],
            "decoder_input_mask": self.decoder_input_mask[index],
            "decoder_target_mask": self.decoder_target_mask[index],
        }


@dataclass
class TranslationView:
    """指定翻译方向后的 Seq2Seq 数据视图。

    数据字段（如 ``source_ids``、``decoder_target_ids``）通过属性转发到
    ``data``，同时携带当前方向对应的 source/target 词表。
    """

    data: Seq2SeqSplit
    source_vocab: Vocabulary
    target_vocab: Vocabulary
    direction: str

    def __getattr__(self, name: str):
        return getattr(self.data, name)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        return self.data[index]


@dataclass
class NumberSeq2SeqDataset:
    """数字到中文数字的 Seq2Seq 数据集。

    ``train`` 和 ``test`` 都是 :class:`Seq2SeqSplit`。其中：

    * ``source_ids``：阿拉伯数字字符，末尾带 ``<EOS>``；
    * ``target_ids``：``<SOS>`` + 中文数字 + ``<EOS>``；
    * ``decoder_input_ids``：给解码器的输入，``<SOS>`` + 中文数字；
    * ``decoder_target_ids``：teacher forcing 的标签，中文数字 + ``<EOS>``。
    """

    train: Seq2SeqSplit
    test: Seq2SeqSplit
    source_vocab: Vocabulary
    target_vocab: Vocabulary
    random_state: int
    min_value: int
    max_value: int
    shared_vocab: bool

    @property
    def src_vocab(self) -> Vocabulary:
        """source_vocab 的常用缩写。"""
        return self.source_vocab

    @property
    def tgt_vocab(self) -> Vocabulary:
        """target_vocab 的常用缩写。"""
        return self.target_vocab

    @property
    def vocab(self) -> dict:
        """以字典形式同时获取两个词表。"""
        return {"src": self.source_vocab, "tgt": self.target_vocab}

    @property
    def vocab_size(self) -> int:
        """共享词表大小；独立词表模式下返回 source 词表大小。"""
        return len(self.source_vocab)

    @staticmethod
    def _normalize_token_type(token_type: str) -> str:
        """统一“阿拉伯/中文”等调用写法。"""
        if not isinstance(token_type, str):
            raise TypeError("token_type 必须是字符串，例如 '阿拉伯' 或 '中文'")
        aliases = {
            "arabic": "arabic",
            "arabic_number": "arabic",
            "src": "arabic",
            "source": "arabic",
            "阿拉伯": "arabic",
            "阿拉伯数字": "arabic",
            "输入": "arabic",
            "中文": "chinese",
            "中文数字": "chinese",
            "chinese": "chinese",
            "target": "chinese",
            "tgt": "chinese",
            "输出": "chinese",
        }
        normalized = aliases.get(token_type.strip().lower())
        if normalized is None:
            raise ValueError(
                "token_type 只能是 '阿拉伯'/'arabic' 或 '中文'/'chinese'"
            )
        return normalized

    @staticmethod
    def _normalize_split(split: str) -> str:
        """统一“训练集/测试集”等调用写法。"""
        if not isinstance(split, str):
            raise TypeError("split 必须是字符串，例如 '训练集' 或 'test'")
        aliases = {
            "train": "train",
            "training": "train",
            "训练": "train",
            "训练集": "train",
            "test": "test",
            "testing": "test",
            "测试": "test",
            "测试集": "test",
        }
        normalized = aliases.get(split.strip().lower())
        if normalized is None:
            raise ValueError(
                "split 只能是 '训练集'/'train' 或 '测试集'/'test'"
            )
        return normalized

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        """统一双向翻译方向的调用写法。"""
        if not isinstance(direction, str):
            raise TypeError(
                "direction 必须是字符串，例如 '阿拉伯->中文' 或 '中文->阿拉伯'"
            )
        aliases = {
            "arabic_to_chinese": "arabic_to_chinese",
            "ar2zh": "arabic_to_chinese",
            "阿拉伯->中文": "arabic_to_chinese",
            "阿拉伯到中文": "arabic_to_chinese",
            "阿拉伯转中文": "arabic_to_chinese",
            "正向": "arabic_to_chinese",
            "chinese_to_arabic": "chinese_to_arabic",
            "zh2ar": "chinese_to_arabic",
            "中文->阿拉伯": "chinese_to_arabic",
            "中文到阿拉伯": "chinese_to_arabic",
            "中文转阿拉伯": "chinese_to_arabic",
            "反向": "chinese_to_arabic",
        }
        normalized = aliases.get(direction.strip().lower())
        if normalized is None:
            raise ValueError(
                "direction 只能是 '阿拉伯->中文'/'ar2zh' 或 "
                "'中文->阿拉伯'/'zh2ar'"
            )
        return normalized

    def get_token_sequences(
        self,
        token_type: str = "阿拉伯",
        split: str = "训练集",
        padded: bool = False,
    ) -> Union[List[List[int]], np.ndarray]:
        """获取训练集或测试集的 token id 序列。

        ``token_type`` 支持 ``"阿拉伯"``/``"arabic"`` 和
        ``"中文"``/``"chinese"``；``split`` 支持 ``"训练集"``/``"train"``
        和 ``"测试集"``/``"test"``。默认返回未补齐的 Python 列表，方便查看
        每条真实序列；设置 ``padded=True`` 时返回右侧 padding 后的二维数组。

        阿拉伯序列末尾包含 ``<EOS>``，中文序列为
        ``<SOS> + 中文数字 + <EOS>``。
        """
        normalized_type = self._normalize_token_type(token_type)
        normalized_split = self._normalize_split(split)
        data_split = getattr(self, normalized_split)

        if normalized_type == "arabic":
            ids = data_split.source_ids
            lengths = data_split.source_lengths
        else:
            ids = data_split.target_ids
            lengths = data_split.target_lengths

        if padded:
            return ids.copy()
        return [row[: int(length)].tolist() for row, length in zip(ids, lengths)]

    def get_translation_data(
        self,
        direction: str = "阿拉伯->中文",
        split: str = "训练集",
    ) -> TranslationView:
        """获取指定方向的完整 Seq2Seq 数据。

        ``direction="阿拉伯->中文"`` 时，source 是阿拉伯数字、target 是中文；
        ``direction="中文->阿拉伯"`` 时两者自动交换。返回对象包含：

        * ``source_ids``：encoder 输入，末尾有 ``<EOS>``；
        * ``decoder_input_ids``：decoder 输入，以 ``<SOS>`` 开头；
        * ``decoder_target_ids``：训练标签，以 ``<EOS>`` 结尾；
        * ``source_vocab`` 和 ``target_vocab``：当前方向对应的词表。

        例如：

        .. code-block:: python

            train = dataset.get_translation_data("中文->阿拉伯", "训练集")
            x = train.source_ids
            decoder_input = train.decoder_input_ids
            y = train.decoder_target_ids
        """
        normalized_direction = self._normalize_direction(direction)
        normalized_split = self._normalize_split(split)

        if normalized_direction == "arabic_to_chinese":
            source_vocab = self.source_vocab
            target_vocab = self.target_vocab
        else:
            source_vocab = self.target_vocab
            target_vocab = self.source_vocab

        all_numbers = np.concatenate((self.train.numbers, self.test.numbers))
        all_source_text, all_target_text = _translation_texts(
            all_numbers, normalized_direction
        )
        all_source_sequences = [
            source_vocab.encode(text, add_eos=True) for text in all_source_text
        ]
        all_target_sequences = [
            target_vocab.encode(text, add_sos=True, add_eos=True)
            for text in all_target_text
        ]
        source_max_length = max(map(len, all_source_sequences))
        target_max_length = max(map(len, all_target_sequences))

        data_split = getattr(self, normalized_split)
        data = _make_translation_split(
            data_split.numbers,
            normalized_direction,
            source_vocab,
            target_vocab,
            source_max_length,
            target_max_length,
        )
        return TranslationView(
            data=data,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            direction=normalized_direction,
        )

    def text_to_token_sequence(
        self,
        text: Union[str, int],
        token_type: str = "阿拉伯",
        add_special_tokens: bool = True,
        as_target: Optional[bool] = None,
    ) -> List[int]:
        """将单条文本转换为 token id 序列。

        例如 ``dataset.text_to_token_sequence("1234")`` 会返回阿拉伯数字
        字符对应的 id，并在末尾添加 ``<EOS>``。当 ``token_type="中文"``
        时也可以传入 ``"一千二百三十四"``；若传入数字字符串，会先转换成
        中文数字再编码。``as_target`` 用于双向翻译时区分 encoder 输入和
        decoder 目标：目标序列包含 ``<SOS>`` 和 ``<EOS>``，encoder 输入只在
        末尾添加 ``<EOS>``。
        """
        normalized_type = self._normalize_token_type(token_type)
        if isinstance(text, (int, np.integer)) and not isinstance(text, bool):
            text = str(int(text))
        if not isinstance(text, str) or not text:
            raise TypeError("text 必须是非空字符串或整数")
        if as_target is not None and not isinstance(as_target, bool):
            raise TypeError("as_target 必须是 bool 或 None")
        if as_target is None:
            # 保留原有调用习惯：阿拉伯默认为 source，中文默认为 target。
            as_target = normalized_type == "chinese"

        if normalized_type == "arabic":
            if any(character not in "0123456789" for character in text):
                raise ValueError("阿拉伯输入只能包含 0 到 9")
            vocab = self.source_vocab
            add_sos = add_special_tokens and as_target
            add_eos = add_special_tokens
        else:
            if text.isdigit():
                text = number_to_chinese(int(text))
            unknown_characters = set(text) - set(
                CHINESE_NUMBER_DIGITS + CHINESE_NUMBER_UNITS
            )
            if unknown_characters:
                raise ValueError(
                    "中文输入包含不支持的字符: "
                    + "".join(sorted(unknown_characters))
                )
            vocab = self.target_vocab
            add_sos = add_special_tokens and as_target
            add_eos = add_special_tokens
        return vocab.encode(text, add_sos=add_sos, add_eos=add_eos)

    # 更直观的单条序列接口别名：dataset.get_token_sequence("1234")。
    get_token_sequence = text_to_token_sequence


def _translation_texts(
    numbers: Sequence[int],
    direction: str,
) -> Tuple[List[str], List[str]]:
    """根据方向生成 source_text 和 target_text。"""
    arabic_text = [str(int(number)) for number in numbers]
    chinese_text = [number_to_chinese(int(number)) for number in numbers]
    if direction == "arabic_to_chinese":
        return arabic_text, chinese_text
    if direction == "chinese_to_arabic":
        return chinese_text, arabic_text
    raise ValueError(f"不支持的翻译方向: {direction}")


def _make_translation_split(
    numbers: np.ndarray,
    direction: str,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    source_max_length: int,
    target_max_length: int,
) -> Seq2SeqSplit:
    source_text, target_text = _translation_texts(numbers, direction)
    source_sequences = [
        source_vocab.encode(text, add_eos=True) for text in source_text
    ]
    target_sequences = [
        target_vocab.encode(text, add_sos=True, add_eos=True) for text in target_text
    ]
    decoder_inputs = [
        target_vocab.encode(text, add_sos=True) for text in target_text
    ]
    decoder_targets = [
        target_vocab.encode(text, add_eos=True) for text in target_text
    ]

    return Seq2SeqSplit(
        numbers=numbers.astype(np.int64, copy=False),
        source_text=source_text,
        target_text=target_text,
        source_ids=_pad_sequences(
            source_sequences, source_vocab.pad_id, source_max_length
        ),
        target_ids=_pad_sequences(
            target_sequences, target_vocab.pad_id, target_max_length
        ),
        decoder_input_ids=_pad_sequences(
            decoder_inputs, target_vocab.pad_id, target_max_length
        ),
        decoder_target_ids=_pad_sequences(
            decoder_targets, target_vocab.pad_id, target_max_length
        ),
        source_lengths=np.asarray([len(sequence) for sequence in source_sequences]),
        target_lengths=np.asarray([len(sequence) for sequence in target_sequences]),
    )


def _make_seq2seq_split(
    numbers: np.ndarray,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    source_max_length: int,
    target_max_length: int,
) -> Seq2SeqSplit:
    source_text = [str(int(number)) for number in numbers]
    target_text = [number_to_chinese(int(number)) for number in numbers]

    source_sequences = [
        source_vocab.encode(text, add_eos=True) for text in source_text
    ]
    target_sequences = [
        target_vocab.encode(text, add_sos=True, add_eos=True) for text in target_text
    ]
    decoder_inputs = [
        target_vocab.encode(text, add_sos=True) for text in target_text
    ]
    decoder_targets = [
        target_vocab.encode(text, add_eos=True) for text in target_text
    ]

    return Seq2SeqSplit(
        numbers=numbers.astype(np.int64, copy=False),
        source_text=source_text,
        target_text=target_text,
        source_ids=_pad_sequences(
            source_sequences, source_vocab.pad_id, source_max_length
        ),
        target_ids=_pad_sequences(
            target_sequences, target_vocab.pad_id, target_max_length
        ),
        decoder_input_ids=_pad_sequences(
            decoder_inputs, target_vocab.pad_id, target_max_length
        ),
        decoder_target_ids=_pad_sequences(
            decoder_targets, target_vocab.pad_id, target_max_length
        ),
        source_lengths=np.asarray([len(sequence) for sequence in source_sequences]),
        target_lengths=np.asarray([len(sequence) for sequence in target_sequences]),
    )


def generate_number_seq2seq_dataset(
    n_samples: Optional[int] = 10_000,
    min_value: int = 0,
    max_value: int = MAX_CHINESE_NUMBER,
    train_ratio: float = 0.8,
    random_state: int = 42,
    shared_vocab: bool = True,
) -> NumberSeq2SeqDataset:
    """生成阿拉伯数字到中文数字的 Seq2Seq 数据集。

    参数：
        n_samples: 样本数。默认 10000；传入 ``None`` 时使用范围内全部数字。
        min_value: 最小数字，包含该值。
        max_value: 最大数字，包含该值，最多支持到 100000。
        train_ratio: 训练集比例，默认 ``0.8``，即 8:2 划分。
        random_state: 固定随机种子，默认 ``42``。
        shared_vocab: 是否合并 source/target 词表，默认 ``True``。开启后两者
            是同一个词表，包含数字、中文字符和特殊 token。

    数据不重复采样，然后用固定种子打乱并划分训练集和测试集。返回对象中的
    id 数组已经右侧 padding，可以直接转换成模型需要的 batch；padding id
    可通过 ``dataset.source_vocab.pad_id`` 或 ``dataset.target_vocab.pad_id`` 获取。
    """
    if isinstance(min_value, bool) or not isinstance(min_value, (int, np.integer)):
        raise TypeError("min_value 必须是整数")
    if isinstance(max_value, bool) or not isinstance(max_value, (int, np.integer)):
        raise TypeError("max_value 必须是整数")
    min_value, max_value = int(min_value), int(max_value)
    if min_value < 0 or max_value > MAX_CHINESE_NUMBER or min_value > max_value:
        raise ValueError(
            f"数字范围必须满足 0 <= min_value <= max_value <= {MAX_CHINESE_NUMBER}"
        )
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须位于 (0, 1) 范围内")
    if not isinstance(shared_vocab, bool):
        raise TypeError("shared_vocab 必须是 bool")
    if isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise TypeError("random_state 必须是整数")
    random_state = int(random_state)

    population_size = max_value - min_value + 1
    if n_samples is None:
        n_samples = population_size
    elif isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError("n_samples 必须是整数或 None")
    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError("n_samples 至少需要为 2，才能同时生成训练集和测试集")
    if n_samples > population_size:
        raise ValueError(
            "默认不重复采样，因此 n_samples 不能超过指定数字范围的大小"
        )

    rng = np.random.default_rng(random_state)
    numbers = rng.choice(
        population_size, size=n_samples, replace=False
    ).astype(np.int64) + min_value
    split_index = int(n_samples * train_ratio)
    if split_index <= 0 or split_index >= n_samples:
        raise ValueError("当前 n_samples 与 train_ratio 无法同时得到非空训练集和测试集")
    train_numbers = numbers[:split_index]
    test_numbers = numbers[split_index:]

    source_vocab, target_vocab = build_number_vocab(shared=shared_vocab)
    all_source_sequences = [
        source_vocab.encode(str(int(number)), add_eos=True) for number in numbers
    ]
    all_target_sequences = [
        target_vocab.encode(number_to_chinese(int(number)), add_sos=True, add_eos=True)
        for number in numbers
    ]
    source_max_length = max(map(len, all_source_sequences))
    target_max_length = max(map(len, all_target_sequences))

    train = _make_seq2seq_split(
        train_numbers,
        source_vocab,
        target_vocab,
        source_max_length,
        target_max_length,
    )
    test = _make_seq2seq_split(
        test_numbers,
        source_vocab,
        target_vocab,
        source_max_length,
        target_max_length,
    )
    return NumberSeq2SeqDataset(
        train=train,
        test=test,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        random_state=random_state,
        min_value=min_value,
        max_value=max_value,
        shared_vocab=shared_vocab,
    )


def get_seq2seq_data(
    dataset: NumberSeq2SeqDataset,
    direction: str = "arabic_to_chinese",
    split: str = "train",
) -> dict:
    """以训练脚本常用的字典形式获取 Seq2Seq 数据。

    参数：
        dataset: ``generate_number_seq2seq_dataset`` 返回的数据集对象。
        direction: ``arabic_to_chinese`` 或 ``chinese_to_arabic``，也支持中文
            方向名称和 ``ar2zh``/``zh2ar`` 别名。
        split: ``train`` 或 ``test``，也支持 ``训练集``/``测试集``。

    返回字典中的 ``encoder_input_ids``、``decoder_input_ids`` 和
    ``decoder_target_ids`` 已经完成 padding，可直接用于训练；同时返回当前
    方向对应的词表、有效长度和 padding mask。``source_ids`` 是
    ``encoder_input_ids`` 的兼容别名。
    """
    if not isinstance(dataset, NumberSeq2SeqDataset):
        raise TypeError(
            "dataset 必须是 generate_number_seq2seq_dataset 返回的对象"
        )

    data = dataset.get_translation_data(direction=direction, split=split)
    return {
        "direction": data.direction,
        "source_ids": data.source_ids,
        "encoder_input_ids": data.encoder_input_ids,
        "target_ids": data.target_ids,
        "decoder_input_ids": data.decoder_input_ids,
        "decoder_target_ids": data.decoder_target_ids,
        "source_lengths": data.source_lengths,
        "target_lengths": data.target_lengths,
        "decoder_lengths": data.decoder_lengths,
        "encoder_mask": data.source_ids != data.source_vocab.pad_id,
        "decoder_input_mask": data.decoder_input_ids != data.target_vocab.pad_id,
        "decoder_target_mask": data.decoder_target_ids != data.target_vocab.pad_id,
        "source_text": data.source_text,
        "target_text": data.target_text,
        "numbers": data.numbers,
        "source_vocab": data.source_vocab,
        "target_vocab": data.target_vocab,
    }


# 更短的名字，方便在 notebook 或训练脚本中使用。
generate_number_dataset = generate_number_seq2seq_dataset
generate_seq2seq_dataset = generate_number_seq2seq_dataset


def generate_moons_dataset(n_samples=300, noise=0.2, random_state=42, train_ratio=0.8):
    """
    生成月牙形二分类数据集，并按比例切分为训练集和测试集
    :param n_samples: 总样本数
    :param noise: 噪声程度
    :param random_state: 随机种子，保证结果可复现
    :param train_ratio: 训练集占比
    :return: (X_train, Y_train), (X_test, Y_test)
    """
    X, Y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    split_idx = int(train_ratio * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    return (X_train, Y_train), (X_test, Y_test)


def generate_multiclass_dataset(
    n_samples=2000,
    n_features=4,
    n_classes=8,
    test_size=0.2,
    random_state=60,
    cluster_std=1.0,
):
    """生成 ``make_blobs`` 多分类数据，并将标签转换为 One-hot。

    数据使用 KSNet 的行主序布局：特征形状为 ``(样本数, 特征数)``，
    标签形状为 ``(样本数, 类别数)``。

    :return: ``(X_train, Y_train), (X_test, Y_test)``
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 每个类别索引会选取单位矩阵中对应的一行，从而得到 One-hot 标签。
    one_hot_lookup = np.eye(n_classes, dtype=np.float64)
    Y_train = one_hot_lookup[y_train]
    Y_test = one_hot_lookup[y_test]
    return (X_train, Y_train), (X_test, Y_test)
