import json
import ast
import regex
from itertools import pairwise
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.cache = {}

        token_set = set(self.vocab.values())
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0

        # append user-provided special tokens if they are not already in vocab
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in token_set:
                self.vocab[next_id] = token_bytes
                token_set.add(token_bytes)
                next_id += 1

        self.bytes2id = {bts: id for id, bts in self.vocab.items()}
        self.merge_priority = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] = None,
    ):
        def bytes_to_unicode():
            bs = (
                list(range(ord("!"), ord("~") + 1))
                + list(range(ord("¡"), ord("¬") + 1))
                + list(range(ord("®"), ord("ÿ") + 1))
            )
            cs = bs[:]
            n = 0

            for b in range(256):
                if b not in bs:
                    bs.append(b)
                    cs.append(256 + n)
                    n += 1

            cs = [chr(c) for c in cs]
            return dict(zip(bs, cs))

        byte_encoder = bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        def gpt2_str_to_bytes(s: str):
            return bytes(byte_decoder[c] for c in s)

        def to_bytes(x):
            if isinstance(x, bytes):
                return x

            if isinstance(x, list):
                return bytes(x)

            if isinstance(x, int):
                return bytes([x])

            if isinstance(x, str):
                if x.startswith("b'") or x.startswith('b"'):
                    try:
                        y = ast.literal_eval(x)
                        if isinstance(y, bytes):
                            return y
                    except Exception:
                        pass

                try:
                    return gpt2_str_to_bytes(x)
                except Exception:
                    pass

                try:
                    return x.encode("latin-1")
                except UnicodeEncodeError:
                    return x.encode("utf-8")

            raise ValueError(f"Cannot convert {x} to bytes")

        # load vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {}

        if isinstance(raw_vocab, dict):
            for k, v in raw_vocab.items():
                # format: {"0": [0], "1": [1], ...}
                if isinstance(k, str) and k.isdigit():
                    vocab[int(k)] = to_bytes(v)

                # GPT-2 format: {"token": id}
                elif isinstance(v, int):
                    vocab[v] = to_bytes(k)

                else:
                    raise ValueError("Unsupported vocab format")

        elif isinstance(raw_vocab, list):
            # format: [[0, [0]], [1, [1]], ...]
            for item in raw_vocab:
                if isinstance(item, list) and len(item) == 2:
                    idx, token = item
                    vocab[int(idx)] = to_bytes(token)
                else:
                    raise ValueError("Unsupported vocab format")

        else:
            raise ValueError("Unsupported vocab format")

        # load merges
        merges: list[tuple[bytes, bytes]] = []

        with open(merges_filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # GPT-2 merges.txt 第一行通常是版本声明
            if line.startswith("#"):
                continue

            try:
                item = ast.literal_eval(line)
                if len(item) != 2:
                    raise ValueError("Unsupported merges format")
                merges.append((to_bytes(item[0]), to_bytes(item[1])))
                continue
            except Exception:
                pass

            parts = line.split()

            if len(parts) != 2:
                raise ValueError("Unsupported merges format")

            merges.append((to_bytes(parts[0]), to_bytes(parts[1])))

        return cls(vocab, merges, special_tokens)

    def _bpe_merge(self, words: bytes) -> list[bytes]:
        if words in self.cache:
            return self.cache[words]

        wordsbytes = [bytes([x]) for x in words]
        merge_priority = self.merge_priority

        while len(wordsbytes) > 1:
            good_pairs = set(
                (l, r) for l, r in pairwise(wordsbytes)
                if (l, r) in merge_priority
            )

            if not good_pairs:
                break

            best_pair = min(good_pairs, key=lambda x: merge_priority[x])

            # O(1) space implementation
            i = 0
            for x in wordsbytes:
                wordsbytes[i] = x
                i += 1

                if i > 1 \
                    and wordsbytes[i - 2] == best_pair[0] \
                    and wordsbytes[i - 1] == best_pair[1]:
                    wordsbytes[i - 2] += wordsbytes[i - 1]
                    i -= 1

            del wordsbytes[i:]

        self.cache[words] = wordsbytes
        return wordsbytes

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        special_tokens = self.special_tokens
        bytes2id = self.bytes2id

        if special_tokens:
            special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
            special_pattern = "|".join(map(regex.escape, special_tokens_sorted))
            chunks = regex.split(f"({special_pattern})", text)
        else:
            chunks = [text]

        ids = []

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in special_tokens:
                ids.append(bytes2id[chunk.encode("utf-8")])
                continue

            for word in regex.findall(PAT, chunk):
                if not word:
                    continue

                merged_word = self._bpe_merge(word.encode("utf-8"))

                for s in merged_word:
                    ids.append(bytes2id[s])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        all2bytes = b"".join(self.vocab[id] for id in ids)
        return all2bytes.decode("utf-8", errors="replace")
