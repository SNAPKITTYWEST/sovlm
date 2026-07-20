from __future__ import annotations
import click
from rich.console import Console
from pathlib import Path
import sys

console = Console()

MODEL_DIR = Path(".sovlm_model")
DICT_PATH = MODEL_DIR / "dictionary.bin"
NGRAM_PATH = MODEL_DIR / "ngram.bin"
SKIPGRAM_PATH = MODEL_DIR / "skipgram.bin"
INVERTED_PATH = MODEL_DIR / "inverted.bin"
FUZZY_PATH = MODEL_DIR / "fuzzy.bin"


def load_model():
    from sovlm.dictionary import Dictionary
    from sovlm.ngram_index import NGramIndex
    from sovlm.skipgram import SkipGramIndex
    from sovlm.kneser_ney import KneserNey
    from sovlm.fuzzy_prefix import FuzzyPrefixIndex
    from sovlm.inverted_index import InvertedIndex
    from sovlm.blender import Blender
    from sovlm.sampler import Sampler, SystemRNG
    from sovlm.generator import Generator

    dictionary = Dictionary.load(DICT_PATH)
    ngram = NGramIndex.load(NGRAM_PATH)
    skipgram = SkipGramIndex.load(SKIPGRAM_PATH)
    inverted = InvertedIndex.load(INVERTED_PATH)
    fuzzy = FuzzyPrefixIndex()
    fuzzy.build_from_index(ngram)
    kn = KneserNey(ngram)
    sampler = Sampler(rng=SystemRNG(), temperature=0.8)
    blender = Blender(kn, inverted, dictionary, sampler)
    return Generator(dictionary, kn, skipgram, fuzzy, inverted, blender, sampler)


@click.group()
def app():
    """SovLM — Sovereign Statistical Language Model. No training. No GPU. Pure governance."""
    pass


@app.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--order", "-n", default=4, help="N-gram order (default 4)")
@click.option("--chunk-size", default=256, help="Retrieval chunk size")
def build(paths, order, chunk_size):
    """Build model from corpus files (.txt .md .lean .pl .jsonl)."""
    if not paths:
        console.print("[red]Provide corpus paths[/red]"); sys.exit(1)
    MODEL_DIR.mkdir(exist_ok=True)
    from sovlm.corpus import build_from_paths
    console.print(f"[green]Building from {len(paths)} files (order={order})...[/green]")
    d, ng, sg, inv, fz = build_from_paths(
        list(paths), DICT_PATH, NGRAM_PATH, SKIPGRAM_PATH, INVERTED_PATH, FUZZY_PATH,
        ngram_order=order, chunk_size=chunk_size)
    console.print(f"[green]Done.[/green] vocab={len(d)} buckets={len(ng.buckets)} chunks={inv.total_chunks}")


@app.command()
@click.option("--prompt", "-p", default="", help="Prompt")
@click.option("--max-tokens", "-m", default=128)
@click.option("--temperature", "-t", default=0.8)
@click.option("--stream", "do_stream", is_flag=True, help="Stream output token by token")
@click.option("--qrng", is_flag=True, help="Use quantum RNG (ANU)")
def generate(prompt, max_tokens, temperature, do_stream, qrng):
    """Generate text from a prompt."""
    gen = load_model()
    if qrng:
        from sovlm.sampler import QRNGClient
        gen.sampler.rng = QRNGClient()
        console.print("[yellow]QRNG active[/yellow]")
    if do_stream:
        console.print(f"[bold]{prompt}[/bold]", end="")
        for tok in gen.stream(prompt, max_tokens=max_tokens, temperature=temperature):
            console.print(tok + " ", end="")
        console.print()
    else:
        out = gen.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        console.print(f"[bold]{prompt}[/bold]{out}")


@app.command()
@click.option("--port", default=8080)
def serve(port):
    """HTTP API server (FastAPI + uvicorn)."""
    try:
        import uvicorn
        from sovlm.api import create_app
        gen = load_model()
        uvicorn.run(create_app(gen), host="0.0.0.0", port=port)
    except ImportError:
        console.print("[red]Install uvicorn + fastapi: pip install uvicorn fastapi[/red]")
