from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import math
import orjson


@dataclass(slots=True)
class InvertedIndex:
    postings: dict[int, list[tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    chunk_lengths: list[int] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)
    chunk_token_ids: list[list[int]] = field(default_factory=list)
    total_chunks: int = 0
    avg_chunk_len: float = 0.0
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    def add_chunk(self, chunk_id: int, text: str, token_ids: list[int]) -> None:
        self.chunk_texts.append(text)
        self.chunk_token_ids.append(token_ids)
        self.chunk_lengths.append(len(token_ids))
        self.total_chunks += 1
        for tok, count in Counter(token_ids).items():
            self.postings[tok].append((chunk_id, count))
        self.avg_chunk_len = sum(self.chunk_lengths) / self.total_chunks

    def bm25_score(self, query_ids: list[int], chunk_id: int) -> float:
        score = 0.0
        chunk_len = self.chunk_lengths[chunk_id]
        for tok, qf in Counter(query_ids).items():
            postings = self.postings.get(tok, [])
            df = len(postings)
            if df == 0:
                continue
            idf = math.log((self.total_chunks - df + 0.5) / (df + 0.5) + 1.0)
            tf = next((c for cid, c in postings if cid == chunk_id), 0)
            if tf == 0:
                continue
            denom = tf + self.BM25_K1 * (1 - self.BM25_B + self.BM25_B * chunk_len / self.avg_chunk_len)
            score += idf * (tf * (self.BM25_K1 + 1)) / denom
        return score

    def retrieve(self, query_ids: list[int], top_k: int = 5) -> list[tuple[float, int, str]]:
        if not query_ids:
            return []
        candidates = set()
        for tok in query_ids:
            for cid, _ in self.postings.get(tok, []):
                candidates.add(cid)
        scored = sorted([(self.bm25_score(query_ids, cid), cid) for cid in candidates], reverse=True)
        return [(score, cid, self.chunk_texts[cid]) for score, cid in scored[:top_k]]

    def save(self, path: Path) -> None:
        path.write_bytes(orjson.dumps({
            "postings": {str(k): v for k, v in self.postings.items()},
            "chunk_lengths": self.chunk_lengths,
            "chunk_texts": self.chunk_texts,
            "chunk_token_ids": self.chunk_token_ids,
            "total_chunks": self.total_chunks,
            "avg_chunk_len": self.avg_chunk_len,
        }))

    @classmethod
    def load(cls, path: Path) -> InvertedIndex:
        data = orjson.loads(path.read_bytes())
        idx = cls()
        idx.postings = defaultdict(list, {int(k): v for k, v in data["postings"].items()})
        idx.chunk_lengths = data["chunk_lengths"]
        idx.chunk_texts = data["chunk_texts"]
        idx.chunk_token_ids = data["chunk_token_ids"]
        idx.total_chunks = data["total_chunks"]
        idx.avg_chunk_len = data["avg_chunk_len"]
        return idx
