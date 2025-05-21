import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

import regex as re
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import SoftTemporaryDirectory
from torch.nn import functional as F
from tqdm.auto import tqdm, trange

WHITESPACE_SPLITTER = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class ByteLevelBPETokenizer:
    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        eos_token: str = "[EOS]",
    ):
        """Byte-Level BPE Tokenizer

        Args:
            vocab: mapping from string token to id
            merges: list of merges in prioritized order
            eos_token: string representation of EOS token
        """
        super().__init__()
        if eos_token not in vocab:
            raise ValueError("There is no EOS token in vocab")
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.token2id = vocab
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.eos_token = eos_token
        self.eos_token_id = self.token2id[eos_token]

        # The closer the pair is to the beginning, the higher the rank
        self.merges = merges
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

    @lru_cache
    def bpe(self, word: tuple[str]) -> tuple[str]:
        """Process word into tokenized representation.
        Word is a tuple of base tokens, i.e. bytes.

        Under the hood:
        1. Tracks the set of token pairs, bi-grams
        2. While possible, replaces the highest-ranking pair with its union

        Args:
            word: list of base string tokens
        Return:
            list of BPE tokens
        """
        word = [
            self.byte_encoder[byte] for byte in word.encode("utf-8")
        ]  # word.encode('utf-8')
        for merge in self.merges:
            tokens = []
            word_unchanged, prev_add = True, False
            for i in range(len(word)):
                if word[i] == merge[0] and word[min(len(word) - 1, i + 1)] == merge[1]:
                    tokens.append(merge[0] + merge[1])
                    word_unchanged, prev_add = False, True
                else:
                    if not prev_add:
                        tokens.append(word[i])
                    prev_add = False
            word = tokens
        ids = []
        for token in word:
            ids.append(self.token2id[token])
        return ids

    def encode(self, text: str, add_eos_token: bool = True) -> list[int]:
        """Convert string to list of token ids.

        Args:
            text: input string, may contain multiple words
            add_eos_token: whether to add eos token id at the end
        Return:
            list of ints, ids of tokenized text
        """
        words = WHITESPACE_SPLITTER.findall(text)
        tokens = []
        for word in words:
            tokens.extend(self.bpe(word))
        if add_eos_token:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, idx: list[int]) -> str:
        """Convert list of tokens' ids to text, opposite to encode method

        Args:
            idx: list of tokens' ids
        Return:
            string, decoded text
        """
        text = ""
        raw_bytes = []
        for id in idx:
            if id == self.eos_token_id:
                break
            tokens = tuple(self.id2token[id])
            raw_bytes.extend([self.byte_decoder[token] for token in tokens])
        text = bytes(raw_bytes).decode("utf-8", errors="replace")
        return text.strip()

    def push_to_hub(self, repo_id, *, private=None, token=None):
        api = HfApi()
        repo_id = api.create_repo(
            repo_id=repo_id, token=token, private=private, exist_ok=True
        ).repo_id

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp:
            save_directory = Path(tmp) / repo_id
            save_directory.mkdir(parents=True)
            with open(save_directory / "vocabulary.json", "w") as f_out:
                print(json.dumps(self.token2id, indent=2), file=f_out)
            with open(save_directory / "merges.json", "w") as f_out:
                print(json.dumps({"merges": self.merges}), file=f_out)

            return api.upload_folder(
                repo_id=repo_id, folder_path=save_directory, token=token
            )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *, token=None, **model_kwargs
    ):
        if not os.path.isdir(pretrained_model_name_or_path):
            storage_folder = snapshot_download(
                repo_id=pretrained_model_name_or_path, token=token
            )
        else:
            storage_folder = pretrained_model_name_or_path
        storage_folder = Path(storage_folder)
        with open(storage_folder / "vocabulary.json", "r") as f_in:
            vocab = json.load(f_in)
        with open(storage_folder / "merges.json", "r") as f_in:
            merges = [tuple(it) for it in json.load(f_in)["merges"]]
        return cls(vocab, merges, **model_kwargs)


def bytes_to_unicode() -> dict[int, str]:
    """The original dictionary consists of 256 bytes and their corresponding Unicode characters.
    For example, chr(33) is '!'. However, not all bytes have a visually appealing representation,
    so such characters are skipped and replaced with the first available ones, i.e. shifted by 256.
    """
    initial_bytes = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    initial_chars = [chr(it) for it in initial_bytes]
    n = 0
    for byte in range(2**8):
        if byte not in initial_bytes:
            initial_bytes.append(byte)
            initial_chars.append(chr(2**8 + n))
            n += 1
    return dict(sorted(zip(initial_bytes, initial_chars)))


def merge(
    merge_pair: tuple[str, str],
    pair_frequences: Counter[tuple[str, str]],
    words_by_tokens: Counter[tuple[str]],
):
    """Merges a given pair of tokens and update corresponding stats

    Args:
        merge_pair: The pair of tokens to be merged.
        pair_frequences: A counter tracking the frequency of token pairs in the dataset.
        words_by_tokens: A counter mapping tokenized words to their frequencies.

    Returns:
        Updated pair frequences and word tokenization w.r.t. to new token.
    """
    updated_words_by_tokens = Counter()
    for tokens, word_freq in words_by_tokens.items():
        word_unchanged, prev_add = True, False
        updated_tokens = []
        for i in range(len(tokens)):
            if (
                tokens[i] == merge_pair[0]
                and tokens[min(i + 1, len(tokens) - 1)] == merge_pair[1]
            ):
                if i + 2 < len(tokens):
                    pair_frequences[
                        (merge_pair[0] + merge_pair[1], tokens[i + 2])
                    ] += word_freq
                if i >= 1:
                    pair_frequences[
                        (tokens[i - 1], merge_pair[0] + merge_pair[1])
                    ] += word_freq
                updated_tokens.append(merge_pair[0] + merge_pair[1])
                # updated_words_by_tokens[updated_tokens] += word_freq
                word_unchanged, prev_add = False, True
            else:
                if not prev_add:
                    updated_tokens.append(tokens[i])
                prev_add = False
        if word_unchanged:
            updated_words_by_tokens[tokens] += word_freq
        else:
            updated_words_by_tokens[tuple(updated_tokens)] += word_freq
    assert len(updated_words_by_tokens) == len(
        words_by_tokens
    ), f"{len(updated_words_by_tokens)}, {len(words_by_tokens)}"
    return pair_frequences, updated_words_by_tokens


def train_tokenizer(
    data: list[str], vocab_size: int = 1024, special_tokens: list[str] = None
):
    """Train BPE tokenizer on passed data

    Args:
        data: List of train documents
        vocab_size: Size of target vocabulary
        special_tokens: List of special tokens to add into vocabulary
    Returns:
        vocabulary: mapping from string token to id
        merges: list of merges, each one is tuple of string tokens
    """
    if vocab_size < 256:
        raise ValueError("Vocab size can't be less than 256")
    if special_tokens is None:
        special_tokens = []

    # 1. Initialize vocabulary (using inverse one during training)
    id2token = bytes_to_unicode()
    merges = []

    # 2. Load data
    words_by_tokens = Counter()
    for sample in tqdm(data, desc="Loading data"):
        # 2.1 Split into words
        words = WHITESPACE_SPLITTER.findall(
            sample.strip()
        )  # list(map(lambda s: s.strip(), WHITESPACE_SPLITTER.findall(sample.strip())))
        for word in words:
            # 2.2 Tokenize with base vocabulary
            tokens = tuple([id2token[byte] for byte in word.encode("utf-8")])
            words_by_tokens[tokens] += 1

    # 3. Calculate statistic of token's pairs
    pair_frequences = Counter()
    tokens = list(words_by_tokens.keys())
    for word_tokens, word_freq in words_by_tokens.items():
        for i in range(len(word_tokens) - 1):
            pair_frequences[(word_tokens[i], word_tokens[i + 1])] += word_freq

    # 4. Build vocabulary
    pbar = trange(
        vocab_size,
        desc="Building vocabulary",
        initial=len(id2token) + len(special_tokens),
    )
    while len(id2token) < vocab_size - len(special_tokens):
        if len(pair_frequences) == 0:
            print("Not enough data to fulfil vocabulary")
            break

        # 4.1 Find the most frequent pair and create new token
        top_pair = pair_frequences.most_common(1)[0][0]
        new_token = "".join(top_pair[0] + top_pair[1])
        del pair_frequences[top_pair]

        # 4.2 Add to vocabulary
        if new_token in id2token.values():
            continue
        id2token[len(id2token)] = new_token
        merges.append(top_pair)

        # 4.3 Update stats and merge the top pair in all tokens
        pair_frequences, words_by_tokens = merge(
            top_pair, pair_frequences, words_by_tokens
        )
        pbar.update()
    pbar.close()

    # 5. Add special tokens
    for special_token in special_tokens:
        id2token[len(id2token)] = special_token

    return {v: k for k, v in id2token.items()}, merges
