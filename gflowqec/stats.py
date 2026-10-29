"""The interval statistics the paper prints (the statistical procedures of the appendix and their table).

    katz       ratio of two rates with its log-ratio interval (Katz); used at n = 180.
    paired_t   paired t interval and two-sided p over seeds, in per cent of the second method's mean; used at n = 300.
    wilson     Wilson score interval on a failure count, for a best code's re-estimated error rate.
    pq         per-qubit logical error rate of a code, applied to a Wilson interval's ends.
"""
import math

import numpy as np
from scipy import stats


def katz(a, na, b, nb):
    """Ratio of the rates a / na and b / nb with its 95 % Katz log interval.

    a and b are counts (for example certified top-scoring codes) out of na and nb paid calls. Returns
    (ratio, low, high). The interval needs both counts nonzero; with a zero count the ratio is returned
    (infinite when b = 0) and both ends are nan.
    """
    if a == 0 or b == 0:
        return (a / na) / (b / nb) if b else float("inf"), float("nan"), float("nan")
    r = (a / na) / (b / nb)
    # standard error of log(r); the interval is r * exp(-/+ 1.96 se)
    se = np.sqrt(1 / a - 1 / na + 1 / b - 1 / nb)
    return r, r * np.exp(-1.96 * se), r * np.exp(1.96 * se)


def paired_t(a, b):
    """Compare two methods run on the same seeds: a[s] and b[s] are numpy arrays, one value per seed.

    Returns (pct, low, high, per_seed_pct, wins, p):
      pct           mean paired difference a - b, in per cent of b's mean;
      low, high     its 95 % paired t interval (len(a) - 1 degrees of freedom), same scale;
      per_seed_pct  the per-seed differences a - b, same scale;
      wins          number of seeds where a is higher than b;
      p             two-sided p-value of the paired t test (null: mean difference zero).
    """
    diff = a - b
    # half-width of the 95 % interval: t quantile times the standard error of the mean difference
    half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / math.sqrt(len(diff))
    ref = b.mean()
    # the t statistic and its two-sided tail probability
    se = diff.std(ddof=1) / math.sqrt(len(diff))
    t = diff.mean() / se
    p = 2 * stats.t.sf(abs(t), len(diff) - 1)
    return (100 * diff.mean() / ref, 100 * (diff.mean() - half) / ref, 100 * (diff.mean() + half) / ref,
            100 * diff / ref, int((diff > 0).sum()), p)


def pq(ler, k):
    """Per-qubit logical error rate of a code with k logical qubits whose logical error rate is ler:
    1 - (1 - ler)^(1/k)."""
    return 1.0 - (1.0 - ler) ** (1.0 / k)


def wilson(f, n):
    """95 % Wilson score interval on the rate f / n (f failures in n shots). Returns (low, high)."""
    z = 1.96                                   # two-sided 95 % normal quantile
    p = f / n
    den = 1 + z*z/n
    c = p + z*z/(2*n)                          # centre of the interval, before dividing by den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))   # half-width, before dividing by den
    return (c - h)/den, (c + h)/den
