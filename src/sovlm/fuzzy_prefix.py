from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import xxhash
import numpy as np


@dataclass(slots=True)
class FuzzyPrefixIndex:
    num_hashes: int = 16
    num_bands: int = 4
    band_size: int = 4
    buckets: dict[int, list[tuple[int, ...]]] = field(default_factory=lambda: defaultdict(list))
    prefixes: set[tuple[int, ...]] = field(default_factory=set)

    def _seeds(self) -> np.ndarray:
        return np.arange(self.num_hashes, dtype=np.uint64) * 0x9e3779b97f4a7c15 + 0xbf58476d1ce4e5b9

    def _signature(self, prefix: tuple[int, ...]) -> np.ndarray:
        seeds = self._seeds()
        sig = np.full(self.num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
        for pos, token in enumerate(prefix):
            for h in range(self.num_hashes):
                val = xxhash.xxh64(token.to_bytes(8, 'little', signed=False), seed=int(seeds[h] + pos)).intdigest()
                if val < sig[h]:
                    sig[h] = val
        return sig

    def _band_key(self, sig: np.ndarray, band: int) -> int:
        start = band * self.band_size
        key = 0
        for v in sig[start:start + self.band_size]:
            key = (key * 0x9e3779b97f4a7c15 + v) & 0xFFFFFFFFFFFFFFFF
        return key

    def add(self, prefix: tuple[int, ...]) -> None:
        if prefix in self.prefixes:
            return
        self.prefixes.add(prefix)
        sig = self._signature(prefix)
        for band in range(self.num_bands):
            self.buckets[self._band_key(sig, band)].append(prefix)

    def query(self, prefix: tuple[int, ...], max_candidates: int = 50) -> list[tuple[int, ...]]:
        sig = self._signature(prefix)
        candidates: set[tuple[int, ...]] = set()
        for band in range(self.num_bands):
            candidates.update(self.buckets.get(self._band_key(sig, band), []))
        scored = sorted([(np.sum(sig == self._signature(c)) / self.num_hashes, c) for c in candidates], reverse=True)
        return [c for _, c in scored[:max_candidates]]

    def build_from_index(self, ngram_index) -> None:
        for prefix in ngram_index.buckets.keys():
            self.add(prefix)
