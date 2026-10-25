"""Symmetric Tardos scores with a predeclared conservative cutoff.

Small-demo finite length, not a proof of arbitrary coalition resistance.
The Bernstein cutoff bounds innocent scores under the marking/independence
model; experiments report their own denominators and Wilson intervals.
"""
import hashlib
import hmac
import math
import numpy as np


class Tardos:
    def __init__(self, length=256, seed=2026, n=5, epsilon=.01, delta=.05):
        self.length, self.n, self.epsilon = length, n, epsilon
        rng = np.random.default_rng(seed)
        low = math.asin(math.sqrt(delta))
        self.p = np.sin(rng.uniform(low, math.pi / 2 - low, length)) ** 2
        self.master = hashlib.sha256(f'TVMC/demo-only/{seed}'.encode()).digest()
        # Fixed before observing any attack. Sum-score (not normalized-score) bound.
        a = math.log(n / epsilon)
        b = math.sqrt((1 - delta) / delta)
        self.threshold = b * a / 3 + math.sqrt(2 * length * a + (b * a / 3) ** 2)

    def encode(self, seed):
        values = [int.from_bytes(hmac.new(self.master,
                  f'code/{seed}/{k}'.encode(), hashlib.sha256).digest()[:8], 'big') / 2**64
                  for k in range(self.length)]
        return (np.array(values) < self.p).astype(np.int8)

    def scores(self, words, observed):
        observed, words = np.asarray(observed), np.asarray(words)
        if observed.shape != (self.length,) or not np.isin(observed, [-1, 0, 1]).all():
            raise ValueError('invalid observation shape/symbol')
        match = np.where(words == 1, np.sqrt((1 - self.p) / self.p),
                         np.sqrt(self.p / (1 - self.p)))
        wrong = np.where(words == 1, -np.sqrt((1 - self.p) / self.p),
                         -np.sqrt(self.p / (1 - self.p)))
        return np.where(observed < 0, 0, np.where(words == observed, match, wrong)).sum(axis=-1)

    def decode(self, words, observed):
        return np.flatnonzero(self.scores(words, observed) >= self.threshold).tolist()


def wilson(success, total):
    if not total:
        return [None, None]
    z, p = 1.959963984540054, success / total
    center = (p + z*z/(2*total))/(1+z*z/total)
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [max(0, center-radius), min(1, center+radius)]


def collude(a, b, strategy, rng):
    if strategy == 'interleave':
        return np.where(rng.integers(0, 2, len(a)), a, b)
    if strategy == 'and':
        return a & b
    if strategy == 'or':
        return a | b
    raise ValueError(strategy)
