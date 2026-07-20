from sovlm.dictionary import Dictionary
from sovlm.inverted_index import InvertedIndex
from sovlm.corpus import tokenize


def test_bm25_retrieval():
    d = Dictionary()
    inv = InvertedIndex()

    chunks = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "quantum mechanics governs atomic transitions",
    ]
    for i, text in enumerate(chunks):
        ids = [d.add(tok) for tok in tokenize(text)]
        inv.add_chunk(i, text, ids)
    d.freeze()

    query = [d["the"], d["cat"]]
    results = inv.retrieve(query, top_k=2)
    assert len(results) > 0
    assert results[0][1] == 0  # chunk 0 has "cat"
