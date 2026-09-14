#!/usr/bin/env python3
"""Binomial contrasts and pooling for the governor experiments, defined once.

Every governor number in this project is a rate over independent episodes, every claim is a
difference of two such rates, and every headline is those differences pooled over densities and
evaluation seeds. That arithmetic had been retyped in each summary script, which is how a progress
view and a final table start disagreeing about the same cell. This module is the single definition.

Counts are the input, never rates: `docs/plans/confirmation_phase_plan_2026-09-06.md` §4 freezes
"every table carries counts/n and derives the rate from them".

CPU-only, standard library only, no simulator import.
"""

from __future__ import annotations

import math

Z95 = 1.959963984540054

# outcome rate field -> the count field the evaluator writes beside it
COUNT_FIELD = {"crash_rate": "crash", "capture_rate": "captured", "timeout_rate": "timeout"}


def outcome_count(outcome, field):
    """The integer count behind an outcome rate, preferring the recorded count over rate x n."""
    name = COUNT_FIELD.get(field)
    if name is not None and name in outcome:
        return int(outcome[name])
    raise KeyError(f"no count recorded for {field}; outcome has {sorted(outcome)}")


def wald_diff(a_count, a_total, b_count, b_total, scale=100.0):
    """(delta, se) of pa - pb on the given scale (default percentage points).

    Wald, i.e. the normal approximation with each arm's own variance. Both arms are independent
    evaluations, so the variances add.
    """
    if a_total <= 0 or b_total <= 0:
        raise ValueError("empty arm")
    pa, pb = a_count / a_total, b_count / b_total
    delta = scale * (pa - pb)
    se = scale * math.sqrt(pa * (1.0 - pa) / a_total + pb * (1.0 - pb) / b_total)
    return delta, se


def ci(delta, se, z=Z95):
    return delta - z * se, delta + z * se


def degenerate_se(se):
    """A Wald standard error of zero means the interval is undefined, not infinitely precise.

    p(1-p) is exactly zero at p in {0, 1}, so two arms that both sit at a boundary — no crashes in
    either, or every episode captured — give se == 0.0 exactly. The interval then collapses to a
    point, two_sided_p returns 0.0 and excludes_zero used to return True: a contrast with no
    statistical content reported as infinitely significant, and that flag gates PASS/FAIL in the
    A5/A7/A8 tables and in seed-replication pooling.
    """
    return not (se > 0.0)


def excludes_zero(delta, se, z=Z95):
    if degenerate_se(se):
        # A zero-width interval excludes everything it does not contain, which would make every
        # degenerate contrast "significant". Refuse to claim exclusion instead.
        return False
    lo, hi = ci(delta, se, z)
    return lo > 0.0 or hi < 0.0


def two_sided_p(delta, se):
    if se <= 0.0:
        return 0.0 if delta else 1.0
    return math.erfc(abs(delta) / (se * math.sqrt(2.0)))


def _reject_degenerate(estimates, where):
    """Inverse-variance weighting divides by the standard error; zero is not a small number.

    These three pooling functions used to raise ZeroDivisionError from inside a comprehension,
    which names neither the cell nor the reason. Near-zero is the quieter half of the same
    problem: a cell at 1 crash in 2048 carries 77 times the weight of an ordinary cell and can
    move a pooled effect from -0.98 pp to -0.03 pp on its own. That is a property of the
    estimator, so it is reported rather than refused; only the exactly-degenerate case refuses.
    """
    bad = [index for index, (_, se) in enumerate(estimates) if degenerate_se(se)]
    if bad:
        raise ValueError(
            f"{where}: cell(s) {bad} have a zero or non-finite standard error, which happens when "
            "both arms sit at 0% or 100%. Inverse-variance pooling is undefined there; drop the "
            "cell or report it separately rather than weighting it infinitely."
        )


def pool_fixed(estimates):
    """Inverse-variance fixed-effect pool of (delta, se) pairs -> (delta, se)."""
    estimates = list(estimates)
    if not estimates:
        raise ValueError("nothing to pool")
    _reject_degenerate(estimates, "pool_fixed")
    weights = [1.0 / se ** 2 for _, se in estimates]
    total = sum(weights)
    delta = sum(w * d for w, (d, _) in zip(weights, estimates)) / total
    return delta, math.sqrt(1.0 / total)


def cochran_q(estimates):
    """(Q, df, I^2) heterogeneity of (delta, se) pairs. I^2 is clamped at 0."""
    estimates = list(estimates)
    _reject_degenerate(estimates, "cochran_q")
    centre, _ = pool_fixed(estimates)
    q = sum((d - centre) ** 2 / se ** 2 for d, se in estimates)
    df = len(estimates) - 1
    i2 = max(0.0, (q - df) / q) if q > 0.0 else 0.0
    return q, df, i2


def pool_random(estimates):
    """DerSimonian-Laird random-effect pool -> (delta, se, tau^2).

    Reported next to the fixed-effect pool whenever cells differ by construction (densities are not
    replicates of one another), so a reader can see how much the interval widens once between-cell
    variance is paid for.
    """
    estimates = list(estimates)
    q, df, _ = cochran_q(estimates)
    weights = [1.0 / se ** 2 for _, se in estimates]
    total = sum(weights)
    c = total - sum(w * w for w in weights) / total
    tau2 = max(0.0, (q - df) / c) if c > 0.0 else 0.0
    rw = [1.0 / (se ** 2 + tau2) for _, se in estimates]
    rtotal = sum(rw)
    delta = sum(w * d for w, (d, _) in zip(rw, estimates)) / rtotal
    return delta, math.sqrt(1.0 / rtotal), tau2


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the order given."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        value = (len(pvalues) - rank) * pvalues[index]
        running = max(running, min(1.0, value))
        adjusted[index] = running
    return adjusted


def sign_test_p(successes, trials):
    """Two-sided exact sign test against p = 1/2."""
    if trials <= 0:
        raise ValueError("no trials")
    k = min(successes, trials - successes)
    tail = sum(math.comb(trials, i) for i in range(0, k + 1)) / (2.0 ** trials)
    return min(1.0, 2.0 * tail)


def format_ci(delta, se, digits=2, z=Z95):
    lo, hi = ci(delta, se, z)
    return f"{delta:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


# --- Student t, for when the replication unit is the seed rather than the cell -----------------
#
# Pooling 15 seed x density cells answers "how big is the effect in these cells". It does not
# answer "does it hold for a new evaluation seed", because cells inside one seed share a policy,
# a scene sampler and an RNG stream. The 2026-09-07 external audit made that objection concrete.
# The honest second number treats each seed as ONE observation: mean +/- t(k-1). It is far weaker
# by construction -- three points -- and reporting both is the point, not choosing one.


def _betacf(a, b, x, iterations=300, eps=3e-16):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                    -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + num * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + num / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def betainc_regularised(a, b, x):
    """I_x(a, b), the regularised incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(value, df):
    """P(T > value) for Student's t with `df` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    tail = 0.5 * betainc_regularised(df / 2.0, 0.5, df / (df + value * value))
    return tail if value > 0 else 1.0 - tail


def student_t_ppf(prob, df, lo=-1e3, hi=1e3, eps=1e-12):
    """Inverse CDF by bisection; the CDF is monotone so this is exact to `eps`."""
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1)")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (1.0 - student_t_sf(mid, df)) < prob:
            lo = mid
        else:
            hi = mid
        if hi - lo < eps:
            break
    return 0.5 * (lo + hi)


def seed_level_t(values, confidence=0.95):
    """(mean, se, lo, hi, p, df) treating each element as one independent replicate.

    `values` are per-seed effect estimates in percentage points. With three seeds df is 2 and the
    interval is wide; that width is the honest cost of asking a seed-generalisation question.
    """
    values = list(values)
    k = len(values)
    if k < 2:
        raise ValueError("need at least two replicates")
    mean = sum(values) / k
    var = sum((v - mean) ** 2 for v in values) / (k - 1)
    se = math.sqrt(var / k)
    df = k - 1
    crit = student_t_ppf(0.5 + confidence / 2.0, df)
    p = 2.0 * student_t_sf(abs(mean / se), df) if se > 0 else (0.0 if mean else 1.0)
    return mean, se, mean - crit * se, mean + crit * se, p, df
