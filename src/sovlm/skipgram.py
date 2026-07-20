from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import orjson

DEFAULT_SKIP_PATTERNS = [(-1,), (-2,), (-1, -2), (-1, -3), (-2, -3), (-1, -2, -3), (-1, -2, -4)]


@dataclass(slots=True)
class SkipGramIndex:
    patterns: list[tuple[int, ...]] = field(default_factory=lambda: DEFAULT_SKIP_PATTERNS)
    buckets: dict[tuple[int, ...], Counter] = field(default_factory=lambda: defaultdict(Counter))
    context_counts: Counter = field(default_factory=Counter)

    def add_sequence(self, ids: list[int]) -> None:
        for i in range(len(ids)):
            target = ids[i]
            for pattern in self.patterns:
                parts = []
                valid = True
                for offset in pattern:
                    pos = i + offset
                    if pos < 0 or pos >= len(ids):
                        valid = False
                        break
                    parts.append(ids[pos])
                if not valid:
                    continue
                prefix = tuple(parts)
                self.buckets[prefix][target] += 1
                self.context_counts[prefix] += 1

    def get_bucket(self, prefix: tuple[int, ...]) -> Counter:
        return self.buckets.get(prefix, Counter())

    def has_prefix(self, prefix: tuple[int, ...]) -> bool:
        return prefix in self.buckets

    def vocab_size(self) -> int:
        vocab = set()
        for c in self.buckets.values():
            vocab.update(c.keys())
        return len(vocab)

    def save(self, path: Path) -> None:
        data = {
            "patterns": [list(p) for p in self.patterns],
            "buckets": [[list(k), [[t, c] for t, c in v.items()]] for k, v in self.buckets.items()],
        }
        path.write_bytes(orjson.dumps(data))

    @classmethod
    def load(cls, path: Path) -> SkipGramIndex:
        data = orjson.loads(path.read_bytes())
        idx = cls(patterns=[tuple(p) for p in data["patterns"]])
        for prefix_list, tail_counts in data["buckets"]:
            prefix = tuple(prefix_list)
            idx.buckets[prefix] = Counter({t: c for t, c in tail_counts})
            idx.context_counts[prefix] = sum(c for _, c in tail_counts)
        return idx
