from __future__ import annotations
from collections import Counter
from dataclasses import dataclass


@dataclass(slots=True)
class KneserNey:
    index: object
    discount: float = 0.75

    def _discounted_prob(self, prefix: tuple[int, ...], tail: int) -> float:
        bucket = self.index.get_bucket(prefix)
        if not bucket:
            return 0.0
        total = self.index.context_counts[prefix]
        return max(bucket.get(tail, 0) - self.discount, 0) / total

    def _lambda(self, prefix: tuple[int, ...]) -> float:
        bucket = self.index.get_bucket(prefix)
        if not bucket:
            return 1.0
        total = self.index.context_counts[prefix]
        return (self.discount * len(bucket)) / total

    def _continuation_prob(self, tail: int) -> float:
        total = sum(self.index.continuation_counts.values())
        if total == 0:
            return 1.0 / max(self.index.vocab_size(), 1)
        return self.index.continuation_counts.get(tail, 0) / total

    def prob(self, context: tuple[int, ...], tail: int, temperature: float = 1.0) -> float:
        order = self.index.order
        if len(context) > order - 1:
            context = context[-(order - 1):]
        if len(context) == 0:
            p = self._continuation_prob(tail)
            return p ** (1.0 / temperature) if temperature != 1.0 else p
        if self.index.has_prefix(context):
            p = self._discounted_prob(context, tail) + self._lambda(context) * self.prob(context[1:], tail, temperature)
            return p ** (1.0 / temperature) if temperature != 1.0 else p
        return self.prob(context[1:], tail, temperature)

    def distribution(self, context: tuple[int, ...], temperature: float = 1.0) -> dict[int, float]:
        candidates: set[int] = set()
        ctx = context
        while True:
            if self.index.has_prefix(ctx):
                candidates.update(self.index.get_bucket(ctx).keys())
            if len(ctx) == 0:
                candidates.update(self.index.continuation_counts.keys())
                break
            ctx = ctx[1:]
        probs = {t: self.prob(context, t, temperature) for t in candidates}
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        return probs
