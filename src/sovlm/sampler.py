from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import os
import httpx


class SystemRNG:
    def random(self) -> float:
        return int.from_bytes(os.urandom(8), 'little') / 2**64
    def bytes(self, n: int) -> bytes:
        return os.urandom(n)


class QRNGClient:
    def __init__(self, url: str = "https://qrng.anu.edu.au/API/jsonI.php", api_key: str | None = None):
        self.url = url
        self.api_key = api_key
        self._buffer: bytearray = bytearray()
        self._client = httpx.Client(timeout=10.0)

    def _refill(self, n: int) -> None:
        params = {"length": max(n, 1024), "type": "uint8"}
        if self.api_key:
            params["api_key"] = self.api_key
        resp = self._client.get(self.url, params=params)
        resp.raise_for_status()
        self._buffer.extend(bytes(resp.json()["data"]))

    def random(self) -> float:
        if len(self._buffer) < 8:
            self._refill(1024)
        val = int.from_bytes(self._buffer[:8], 'little')
        self._buffer = self._buffer[8:]
        return val / 2**64

    def bytes(self, n: int) -> bytes:
        while len(self._buffer) < n:
            self._refill(n)
        out = bytes(self._buffer[:n])
        self._buffer = self._buffer[n:]
        return out


@dataclass(slots=True)
class Sampler:
    rng: object
    temperature: float = 1.0
    repetition_penalty: float = 1.1
    decay_window: int = 64
    decay_factor: float = 0.95
    _recent: deque = field(default_factory=deque)
    _penalties: dict[int, float] = field(default_factory=dict)

    def sample(self, probs: dict[int, float]) -> int:
        if not probs:
            return 0
        adjusted = {tok: p / self._penalties.get(tok, 1.0) if self._penalties.get(tok, 1.0) > 1.0
                    else p * self._penalties.get(tok, 1.0) for tok, p in probs.items()}
        if self.temperature != 1.0:
            adjusted = {k: v ** (1.0 / self.temperature) for k, v in adjusted.items()}
        total = sum(adjusted.values())
        if total == 0:
            return max(probs, key=probs.get)
        tokens, weights = zip(*adjusted.items())
        weights = np.array(weights, dtype=np.float64) / total
        r = self.rng.random()
        idx = int(np.searchsorted(np.cumsum(weights), r))
        chosen = tokens[min(idx, len(tokens) - 1)]
        self._recent.append(chosen)
        if len(self._recent) > self.decay_window:
            old = self._recent.popleft()
            self._penalties[old] = max(1.0, self._penalties.get(old, 1.0) * self.decay_factor)
        self._penalties[chosen] = self._penalties.get(chosen, 1.0) * self.repetition_penalty
        return chosen

    def reset(self) -> None:
        self._recent.clear()
        self._penalties.clear()
