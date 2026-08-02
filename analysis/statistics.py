"""Dependency-free correlation and influence diagnostics used by the paper."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from statistics import NormalDist


Correlation = Callable[[Sequence[float], Sequence[float]], float]


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0 or y_var == 0:
        return float("nan")
    return numerator / math.sqrt(x_var * y_var)


def rank_values(values: Sequence[float]) -> list[float]:
    indexed = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][0] == indexed[start][0]:
            end += 1
        average_rank = (start + end - 1) / 2 + 1
        for _, index in indexed[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    return pearson_correlation(rank_values(xs), rank_values(ys))


def kendall_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Kendall's tau-b, including correction for ties on either axis."""

    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    concordant = discordant = ties_x_only = ties_y_only = 0
    for left in range(len(xs) - 1):
        for right in range(left + 1, len(xs)):
            x_sign = (xs[left] > xs[right]) - (xs[left] < xs[right])
            y_sign = (ys[left] > ys[right]) - (ys[left] < ys[right])
            if x_sign == 0 and y_sign == 0:
                continue
            if x_sign == 0:
                ties_x_only += 1
            elif y_sign == 0:
                ties_y_only += 1
            elif x_sign == y_sign:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x_only)
        * (concordant + discordant + ties_y_only)
    )
    if denominator == 0:
        return float("nan")
    return (concordant - discordant) / denominator


def fisher_z_interval(
    correlation: float,
    n: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return the conventional Fisher-z interval for a correlation estimate.

    The transform requires at least four independent model-level observations.
    For the rank coefficients this is an explicitly approximate sensitivity
    interval, reported alongside (not in place of) the coefficient itself.
    """

    if n <= 3 or math.isnan(correlation) or not -1.0 <= correlation <= 1.0:
        return float("nan"), float("nan")
    if correlation == -1.0 or correlation == 1.0:
        return correlation, correlation
    critical = NormalDist().inv_cdf(0.5 + confidence / 2)
    center = math.atanh(correlation)
    margin = critical / math.sqrt(n - 3)
    return math.tanh(center - margin), math.tanh(center + margin)


def leave_group_out_range(
    xs: Sequence[float],
    ys: Sequence[float],
    groups: Sequence[str],
    statistic: Correlation,
) -> tuple[float, float, int]:
    """Range of estimates after omitting each distinct group in turn."""

    if len(xs) != len(ys) or len(xs) != len(groups):
        raise ValueError("xs, ys, and groups must have the same length")
    estimates: list[float] = []
    for omitted in sorted(set(groups)):
        keep = [index for index, group in enumerate(groups) if group != omitted]
        estimate = statistic([xs[index] for index in keep], [ys[index] for index in keep])
        if not math.isnan(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan"), 0
    return min(estimates), max(estimates), len(estimates)

