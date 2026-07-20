from __future__ import annotations
from pathlib import Path
import orjson
import re

from .dictionary import Dictionary
from .ngram_index import NGramIndex
from .skipgram import SkipGramIndex
from .inverted_index import InvertedIndex
from .fuzzy_prefix import FuzzyPrefixIndex


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap) if words[i:i + chunk_size]]


def tokenize(text: str) -> list[str]:
    text = re.sub(r'([.,!?;:()\[\]{}"\'])', r' \1 ', text)
    return text.split()


def build_from_paths(
    paths: list[Path],
    dict_path: Path,
    ngram_path: Path,
    skipgram_path: Path,
    inverted_path: Path,
    fuzzy_path: Path,
    ngram_order: int = 4,
    chunk_size: int = 256,
) -> tuple[Dictionary, NGramIndex, SkipGramIndex, InvertedIndex, FuzzyPrefixIndex]:
    dictionary = Dictionary()
    ngram = NGramIndex(order=ngram_order)
    skipgram = SkipGramIndex()
    inverted = InvertedIndex()
    all_seqs: list[list[int]] = []
    chunk_id = 0

    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix == ".jsonl":
            for line in text.strip().split("\n"):
                try:
                    item = orjson.loads(line)
                    t = item.get("text", "") or item.get("content", "")
                    ids = [dictionary.add(tok) for tok in tokenize(t)]
                    all_seqs.append(ids)
                    for chunk in chunk_text(t, chunk_size):
                        cids = [dictionary.add(tok) for tok in tokenize(chunk)]
                        inverted.add_chunk(chunk_id, chunk, cids); chunk_id += 1
                except Exception:
                    pass
        else:
            ids = [dictionary.add(tok) for tok in tokenize(text)]
            all_seqs.append(ids)
            for chunk in chunk_text(text, chunk_size):
                cids = [dictionary.add(tok) for tok in tokenize(chunk)]
                inverted.add_chunk(chunk_id, chunk, cids); chunk_id += 1

    dictionary.freeze()
    for seq in all_seqs:
        ngram.add_sequence(seq)
        skipgram.add_sequence(seq)

    fuzzy = FuzzyPrefixIndex()
    fuzzy.build_from_index(ngram)

    dictionary.save(dict_path)
    ngram.save(ngram_path)
    skipgram.save(skipgram_path)
    inverted.save(inverted_path)

    return dictionary, ngram, skipgram, inverted, fuzzy
