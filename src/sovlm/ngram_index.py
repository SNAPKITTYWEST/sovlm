from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import orjson


@dataclass(slots=True)
class NGramIndex:
    order: int
    buckets: dict[tuple[int, ...], Counter] = field(default_factory=lambda: defaultdict(Counter))
    context_counts: Counter = field(default_factory=Counter)
    continuation_counts: Counter = field(default_factory=Counter)

    def add_sequence(self, ids: list[int]) -> None:
        if len(ids) < self.order:
            return
        for i in range(len(ids) - self.order + 1):
            prefix = tuple(ids[i:i + self.order - 1])
            tail = ids[i + self.order - 1]
            self.buckets[prefix][tail] += 1
            self.context_counts[prefix] += 1
        self._compute_continuation_counts()

    def _compute_continuation_counts(self) -> None:
        self.continuation_counts.clear()
        for prefix, counter in self.buckets.items():
            for tail in counter:
                self.continuation_counts[tail] += 1

    def get_bucket(self, prefix: tuple[int, ...]) -> Counter:
        return self.buckets.get(prefix, Counter())

    def has_prefix(self, prefix: tuple[int, ...]) -> bool:
        return prefix in self.buckets

    def vocab_size(self) -> int:
        return len(self.continuation_counts)

    def save(self, path: Path) -> None:
        data = {
            "order": self.order,
            "buckets": [[list(k), [[t, c] for t, c in v.items()]] for k, v in self.buckets.items()],
        }
        path.write_bytes(orjson.dumps(data))

    @classmethod
    def load(cls, path: Path) -> NGramIndex:
        data = orjson.loads(path.read_bytes())
        idx = cls(order=data["order"])
        for prefix_list, tail_counts in data["buckets"]:
            prefix = tuple(prefix_list)
            idx.buckets[prefix] = Counter({t: c for t, c in tail_counts})
            idx.context_counts[prefix] = sum(c for _, c in tail_counts)
        idx._compute_continuation_counts()
        return idx
