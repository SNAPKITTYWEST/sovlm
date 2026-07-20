from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class Blender:
    markov: object
    retrieval: object
    dictionary: object
    sampler: object
    min_markov_count: int = 5

    def _markov_confidence(self, context: tuple[int, ...]) -> float:
        bucket = self.markov.index.get_bucket(context) if hasattr(self.markov, 'index') else {}
        total = sum(bucket.values()) if bucket else 0
        return min(1.0, total / (self.min_markov_count * 2))

    def _retrieval_distribution(self, context: tuple[int, ...]) -> dict[int, float]:
        results = self.retrieval.retrieve(list(context), top_k=3)
        if not results:
            return {}
        _, chunk_id, _ = results[0]
        chunk_tokens = self.retrieval.chunk_token_ids[chunk_id]
        if len(chunk_tokens) > len(context):
            return {chunk_tokens[len(context)]: 1.0}
        return {}

    def blended_distribution(self, context: tuple[int, ...], temperature: float) -> dict[int, float]:
        alpha = self._markov_confidence(context)
        p_markov = self.markov.distribution(context, temperature)
        p_retrieval = self._retrieval_distribution(context)
        if not p_retrieval:
            return p_markov
        if not p_markov:
            return p_retrieval
        blended = {}
        for tok in set(p_markov) | set(p_retrieval):
            blended[tok] = alpha * p_markov.get(tok, 0) + (1 - alpha) * p_retrieval.get(tok, 0)
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()} if total > 0 else blended

    def sample_next(self, context: tuple[int, ...], temperature: float = 1.0) -> int:
        return self.sampler.sample(self.blended_distribution(context, temperature))
