"""
Minimal (byte-level) bype pair Encoding Tokenizer
Algorithmically follows the GPT tokenizer (tiktoken)
Supports regex splitting pattern for creating tokens
Supports special tokens
"""

import regex as re
from Tokenizer import Tokenizer, get_stats, merge

# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
# GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[
\r\n]|\s+(?!\S)|\s+"""

GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def train(self, text, vocab_size, verbose=False):

        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunks in ids:
                get_stats(chunks, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i

            ids = [merge(chunks, pair, idx) for chunks in ids]
            merges[pair] = ids
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        # example: {"<|endoftext|>": 100257}

        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _encode_chunk(self, text_byte):

        ids = list(text_byte)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break  # nothing left to merge anymore

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []

        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)

        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        """

        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            return self.encode_ordinary(text)

        # surrounding the pattern with () makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text

        ids = []
        for part in special_chunks:
            if part in special:
                # encode special token separately
                ids.append(special[part])
            else:
                # encode ordinary sequence normally
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):

        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")

        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")

        return text
