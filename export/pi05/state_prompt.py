"""OpenPI's state prompt assembled from exported SentencePiece token pieces."""
import numpy as np


class StatePrompt:
    def __init__(self, model_path):
        import sentencepiece
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        self.sep, self.neg = self.sp.piece_to_id("▁"), self.sp.piece_to_id("▁-")
        self.pieces = {c: self.sp.piece_to_id(c) for c in "0123456789"}
        if self.sp.unk_id() in (self.sep, self.neg, *self.pieces.values()):
            raise ValueError("state prompt digits are not single tokenizer pieces")
        head, full = self.head("x"), self.full("x", [5])
        if full[:len(head) + 2] != head + [self.sep, self.pieces["5"]]:
            raise ValueError("state prompt head is not a prefix of the full tokenizer output")
        self.tail = full[len(head) + 2:]

    @staticmethod
    def clean(task):
        return task.strip().replace("_", " ").replace("\n", " ")

    def head(self, task):
        return self.sp.encode(f"Task: {self.clean(task)}, State:", add_bos=True)

    def full(self, task, bins):
        return self.sp.encode(f"Task: {self.clean(task)}, State: {' '.join(map(str, bins))};\nAction: ", add_bos=True)

    def compose(self, head, bins):
        ids = list(head)
        for value in bins:
            value = int(value)
            ids += [self.neg if value < 0 else self.sep] + [self.pieces[c] for c in str(abs(value))]
        return ids + self.tail

    def verify(self, task, state_dim, seed=0):
        head, rng, checked = self.head(task), np.random.default_rng(seed), 0
        for value in range(-1, 256):
            bins = [value] * state_dim
            if self.compose(head, bins) != self.full(task, bins):
                raise ValueError(f"state prompt assembly differs from tokenizer at bin {value}")
            checked += 1
        for _ in range(300):
            bins = rng.integers(-1, 256, state_dim)
            if self.compose(head, bins) != self.full(task, bins):
                raise ValueError("state prompt assembly differs from tokenizer")
            checked += 1
        return checked
