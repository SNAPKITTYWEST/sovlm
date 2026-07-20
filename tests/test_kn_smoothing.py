from sovlm.dictionary import Dictionary
from sovlm.ngram_index import NGramIndex
from sovlm.kneser_ney import KneserNey


def test_kn_basic():
    d = Dictionary()
    corpus = "the cat sat on the mat the cat ate the rat"
    ids = [d.add(t) for t in corpus.split()]
    d.freeze()

    ng = NGramIndex(order=3)
    ng.add_sequence(ids)
    kn = KneserNey(ng)

    context = (d["the"], d["cat"])
    dist = kn.distribution(context)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_kn_backoff():
    d = Dictionary()
    ids = [d.add(t) for t in "a b c d e".split()]
    d.freeze()
    ng = NGramIndex(order=2)
    ng.add_sequence(ids)
    kn = KneserNey(ng)
    # unseen context should backoff gracefully
    dist = kn.distribution((d["e"],))
    assert isinstance(dist, dict)
