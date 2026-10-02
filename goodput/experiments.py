"""Monte Carlo runs over seeds: compare scenarios and sweep the checkpoint interval."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace

from goodput.analyze import CATEGORIES, build_jobs, summarize
from goodput.checkpoint import daly, expected_goodput, young
from goodput.scenario import Scenario
from goodput.sim import simulate


@dataclass
class Result:
    scenario: Scenario
    seeds: int
    goodput: list[float]
    shares: dict[str, float]  # mean share of GPU-hours per category
    interruptions: float  # mean per run
    mtbi_hours: float  # pooled running hours / interruptions
    lost_gpu_hours: float  # mean unproductive GPU-hours per run

    @property
    def mean(self) -> float:
        return statistics.fmean(self.goodput)

    @property
    def p10(self) -> float:
        return _quantile(self.goodput, 0.1)

    @property
    def p90(self) -> float:
        return _quantile(self.goodput, 0.9)


def run(s: Scenario, seeds: int) -> Result:
    goodputs, shares, ints, running, lost = [], {c: 0.0 for c in CATEGORIES}, 0, 0.0, 0.0
    for seed in range(1, seeds + 1):
        m = summarize(build_jobs(simulate(s, seed)))
        goodputs.append(m.goodput)
        for c in CATEGORIES:
            shares[c] += m.gpu_hours[c] / m.total_gpu_hours / seeds
        ints += m.interruptions
        running += m.running_hours
        lost += (m.total_gpu_hours - m.gpu_hours["productive"]) / seeds
    return Result(s, seeds, goodputs, shares, ints / seeds, running / ints if ints else float("inf"), lost)


def compare(scenarios: list[Scenario], seeds: int) -> list[Result]:
    return [run(s, seeds) for s in scenarios]


@dataclass
class SweepPoint:
    interval_minutes: float
    simulated: float
    p10: float
    p90: float
    model: float


@dataclass
class Sweep:
    scenario: Scenario
    seeds: int
    points: list[SweepPoint]
    young_minutes: float
    daly_minutes: float
    daly_simulated: float

    @property
    def best(self) -> SweepPoint:
        return max(self.points, key=lambda p: p.simulated)


DEFAULT_INTERVALS = (10, 15, 20, 30, 45, 60, 75, 90, 120, 150, 180, 240, 300)


def sweep(s: Scenario, seeds: int, intervals=DEFAULT_INTERVALS) -> Sweep:
    m, c = s.nominal_mtbi_hours(), s.checkpoint_minutes / 60
    overhead = s.mean_detect_hours() + s.restart_minutes / 60
    points = []
    for minutes in intervals:
        r = run(replace(s, checkpoint_interval_minutes=minutes), seeds)
        points.append(SweepPoint(minutes, r.mean, r.p10, r.p90, expected_goodput(m, c, minutes / 60, overhead)))
    d = daly(c, m) * 60
    at_daly = run(replace(s, checkpoint_interval_minutes=d), seeds).mean
    return Sweep(s, seeds, points, young(c, m) * 60, d, at_daly)


def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
