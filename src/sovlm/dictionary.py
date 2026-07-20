from __future__ import annotations
import orjson
from pathlib import Path
from dataclasses import dataclass, field


@dataclass(slots=True)
class Dictionary:
    token2id: dict[str, int] = field(default_factory=dict)
    id2token: list[str] = field(default_factory=list)
    _frozen: bool = False

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

    def __post_init__(self):
        for tok in self.SPECIAL_TOKENS:
            if tok not in self.token2id:
                self._add(tok)

    def _add(self, token: str) -> int:
        if self._frozen:
            raise RuntimeError("Dictionary is frozen")
        idx = len(self.id2token)
        self.token2id[token] = idx
        self.id2token.append(token)
        return idx

    def add(self, token: str) -> int:
        return self.token2id.get(token) if token in self.token2id else self._add(token)

    def __getitem__(self, token: str) -> int:
        return self.token2id.get(token, self.UNK_ID)

    def token(self, idx: int) -> str:
        return self.id2token[idx] if 0 <= idx < len(self.id2token) else "<UNK>"

    def freeze(self) -> None:
        self._frozen = True

    def __len__(self) -> int:
        return len(self.id2token)

    def save(self, path: Path) -> None:
        path.write_bytes(orjson.dumps({"token2id": self.token2id, "id2token": self.id2token}))

    @classmethod
    def load(cls, path: Path) -> Dictionary:
        data = orjson.loads(path.read_bytes())
        d = cls()
        d.token2id = data["token2id"]
        d.id2token = data["id2token"]
        d._frozen = True
        return d

    def encode(self, text: str) -> list[int]:
        return [self[tok] for tok in text.split()]

    def decode(self, ids: list[int]) -> str:
        skip = {self.PAD_ID, self.BOS_ID, self.EOS_ID}
        return " ".join(self.token(i) for i in ids if i not in skip)
