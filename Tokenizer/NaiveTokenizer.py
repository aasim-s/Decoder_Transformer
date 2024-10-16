"""
Minimal (byte-level) bype pair Encoding Tokenizer
Algorithmically follows the GPT tokenizer (tiktoken)
"""

from Tokenizer import Tokenizer, get_stats, merge


class NaiveTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):

        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)

            idx = 256 + i
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:

            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("int")))
            if pair not in self.merges:
                break  # nothing left to merge anymore
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def decode(self, ids):

        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")

        return text

