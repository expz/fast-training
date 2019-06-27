from bleu import bleu_score
from evaluate import moses_bleu_score
import math
import numpy
import random


def test_perfect_bleu():
    hypotheses = ['a b c d']
    references = [['a b c d']]
    bleu = bleu_score(hypotheses, references)
    assert(isinstance(bleu, float))
    assert(bleu == 1.0)


def test_less_than_four_bleu():
    """Moses BLEU is 0 when there are no matching 4-grams."""
    hypotheses = ['a b c']
    references = [['a b c']]
    bleu = bleu_score(hypotheses, references)
    assert(bleu == 0.0)


def test_empty_bleu():
    hypotheses = []
    references = []
    bleu = bleu_score(hypotheses, references)
    assert(bleu == 0.0)


def test_same_size_bleu():
    hypotheses = ['a b c d f']
    references = [['a b c d e']]
    epsilon = 1e-2
    bleu1 = math.exp(
        (math.log(4.0/5) + math.log(3.0/4) + math.log(2.0/3) + math.log(1.0/2))
        / 4)
    bleu2 = bleu_score(hypotheses, references)
    assert(abs(bleu1 - bleu2) < epsilon)


def test_agreement_moses_same_size():
    """
    Test that the perl and python implementations of BLEU
    agree on 100 random pairs of same-sized sequences.
    """
    epsilon = 0.02
    sz = 50
    num_tests = 100
    ref = list(map(str, range(sz)))
    ref_str = ' '.join(ref)
    for n in range(num_tests):
        num_swaps = random.randint(0, 10)
        hyp = list(ref)
        for _ in range(num_swaps):
            i = random.randint(0, sz - 1)
            j = random.randint(0, sz - 1)
            t = hyp[j]
            hyp[j] = hyp[i]
            hyp[i] = t
        hyp_str = ' '.join(hyp)
        bleu1 = bleu_score([hyp_str], [[ref_str]]) * 100.0
        bleu2 = moses_bleu_score([hyp_str], [ref_str])
        assert(abs(bleu1 - bleu2) < epsilon)


def test_test():
    out = "This is my intention and I ask this House to provide the wider support to this report."
    tgt = "That is my intention and I would ask the House to give this report overwhelming support!"
    assert(bleu_score([out], [[tgt]]) == 0.0)

