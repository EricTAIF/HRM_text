# utils/tokenizer_char.py
from typing import Dict, List, Iterable
import json
import os

PAD_ID = 0
EOS_ID = 1  # reserved (unused for now), chars start at 2

class CharTokenizer:
    def __init__(self):
        self.char2id: Dict[str, int] = {}
        self.id2char: List[str] = []

    def fit(self, texts: Iterable[str]) -> None:
        charset = set()
        for t in texts:
            charset.update(list(t))
        # stable order
        vocab = sorted(list(charset))
        # reserve 0=PAD, 1=EOS
        self.char2id = {}
        self.id2char = []
        # indices from 2...
        next_id = 2
        for ch in vocab:
            self.char2id[ch] = next_id
            self.id2char.append(ch)
            next_id += 1

    @property
    def vocab_size(self) -> int:
        # PAD + EOS + chars
        return 2 + len(self.id2char)

    def encode(self, text: str) -> List[int]:
        return [self.char2id[ch] for ch in text if ch in self.char2id]

    def decode(self, ids: List[int]) -> str:
        # inverse: id>=2 maps into id2char[id-2]
        out = []
        for i in ids:
            if i >= 2:
                idx = i - 2
                if 0 <= idx < len(self.id2char):
                    out.append(self.id2char[idx])
        return "".join(out)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"char2id": self.char2id}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r") as f:
            obj = json.load(f)
        tok = cls()
        tok.char2id = obj["char2id"]
        # rebuild id2char
        inv = sorted([(i, ch) for ch, i in tok.char2id.items()], key=lambda x: x[0])
        tok.id2char = [ch for _, ch in inv if _ >= 2]
        return tok
