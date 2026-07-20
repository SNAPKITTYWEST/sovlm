# sovlm

![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)
![Zero GPU](https://img.shields.io/badge/GPU-zero-black?style=flat-square)
![Zero Training](https://img.shields.io/badge/training-zero-black?style=flat-square)
![License](https://img.shields.io/badge/license-SOVEREIGN-gold?style=flat-square)
![Author](https://img.shields.io/badge/author-Ahmad_Ali_Parr-red?style=flat-square)

**Sovereign Statistical Language Model.**
N-grams + Kneser-Ney + BM25 Retrieval + QRNG sampling.
No training. No backprop. No GPU. Pure governance.

Part of the SnapKitty Collective sovereign AI stack.

---

## Install

```bash
pip install -e ".[dev]"
# or
uv pip install -e ".[dev]"
```

## Usage

```bash
# Build from your corpus (Lean proofs, Prolog, docs, anything)
sovlm build data/*.lean data/*.pl data/*.md --order 4

# Generate
sovlm generate -p "theorem main : " -m 200

# Stream
sovlm generate -p "The 49th Call is" --stream

# Quantum RNG sampling
sovlm generate -p "lemma " --qrng

# Serve HTTP API
sovlm serve --port 8080
```

## Python API

```python
from sovlm import Generator, build_from_paths
from pathlib import Path

# Build
d, ng, sg, inv, fz = build_from_paths([Path("corpus.md")], ...)

# Generate
gen = Generator(d, kn, sg, fz, inv, blender, sampler)
result = gen.generate("theorem xor_assoc : ", max_tokens=100)
```

## Architecture

| Component | Role |
|---|---|
| `Dictionary` | Vocabulary sovereignty — you own the tokens |
| `NGramIndex` | Local coherence — what follows what |
| `KneserNey` | Smoothing — handle unseen contexts |
| `SkipGramIndex` | Structural patterns at distance |
| `FuzzyPrefixIndex` | LSH near-miss — graceful degradation |
| `InvertedIndex + BM25` | Retrieval — what's relevant in corpus |
| `Blender` | Arbitration — Markov vs retrieval |
| `Sampler + QRNG` | Entropy — unbiased, quantum-seeded choice |

## SnapKitty Stack Integration

```
sovlm ──► abjad-machine (SUBLEQ address space)
       ──► sovereign-transformer (plasma gate)
       ──► the-49th-call (Lean 4 verified)
       ──► bob-orchestrator (Byzantine council)
```

---

*Ahmad Ali Parr · SnapKitty Collective · 2026*
