from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Generator:
    dictionary: object
    kn: object
    skipgram: object
    fuzzy: object
    inverted: object
    blender: object
    sampler: object

    def generate(self, prompt: str = "", max_tokens: int = 128, temperature: float = 0.8,
                 stop_tokens: set[str] | None = None) -> str:
        stop_ids = {self.dictionary[tok] for tok in (stop_tokens or {"<EOS>"})}
        context_ids = [self.dictionary.BOS_ID] + self.dictionary.encode(prompt)
        generated = []
        order = self.kn.index.order

        for _ in range(max_tokens):
            context = tuple(context_ids[-(order - 1):])
            if not self.kn.index.has_prefix(context) and self.fuzzy.prefixes:
                candidates = self.fuzzy.query(context, max_candidates=5)
                if candidates:
                    dists = [self.kn.distribution(c, temperature) for c in candidates]
                    blended: dict[int, float] = {}
                    for d in dists:
                        for k, v in d.items():
                            blended[k] = blended.get(k, 0) + v
                    total = sum(blended.values())
                    dist = {k: v / total for k, v in blended.items()} if total > 0 else {}
                else:
                    dist = self.kn.distribution(context, temperature)
            else:
                dist = self.kn.distribution(context, temperature)

            if self.inverted.total_chunks > 0:
                next_id = self.blender.sample_next(context, temperature)
            else:
                next_id = self.sampler.sample(dist)

            if next_id in stop_ids:
                break
            generated.append(next_id)
            context_ids.append(next_id)

        return self.dictionary.decode(generated)

    def stream(self, prompt: str = "", **kwargs) -> Iterator[str]:
        stop_ids = {self.dictionary[tok] for tok in kwargs.get("stop_tokens", {"<EOS>"})}
        context_ids = [self.dictionary.BOS_ID] + self.dictionary.encode(prompt)
        order = self.kn.index.order
        temperature = kwargs.get("temperature", 0.8)
        for _ in range(kwargs.get("max_tokens", 128)):
            context = tuple(context_ids[-(order - 1):])
            dist = self.kn.distribution(context, temperature)
            next_id = self.sampler.sample(dist)
            if next_id in stop_ids:
                break
            context_ids.append(next_id)
            yield self.dictionary.token(next_id)
