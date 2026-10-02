"""Checkpoint interval formulas and a first-order goodput model.

Young (1974):  tau = sqrt(2 * C * M)
Daly (2006):   tau = sqrt(2CM) * (1 + 1/3 * sqrt(C / 2M) + 1/9 * (C / 2M)) - C   for C < 2M, else M

tau is compute time between checkpoints, C the time to write one, M the mean time between
interruptions of the job. All in the same unit (hours here). Daly notes that the restart time does
not change the optimum, which is why it is not a parameter.
"""

from __future__ import annotations

import math


def young(checkpoint_h: float, mtbi_h: float) -> float:
    _check(checkpoint_h, mtbi_h)
    return math.sqrt(2 * checkpoint_h * mtbi_h)


def daly(checkpoint_h: float, mtbi_h: float) -> float:
    _check(checkpoint_h, mtbi_h)
    c, m = checkpoint_h, mtbi_h
    if c >= 2 * m:
        return m
    x = c / (2 * m)
    return math.sqrt(2 * c * m) * (1 + math.sqrt(x) / 3 + x / 9) - c


def expected_goodput(mtbi_h: float, checkpoint_h: float, interval_h: float, overhead_h: float) -> float:
    """First-order estimate of goodput (productive time / wall time) for one job.

    Per interruption cycle the job runs for M hours on average, then spends `overhead_h`
    (detection + waiting for nodes + restart) before running again. While running, a share
    C / (tau + C) goes to writing checkpoints, and each interruption throws away on average half a
    checkpoint interval of compute. Good while tau and the overhead are small compared with M.
    """
    _check(checkpoint_h, mtbi_h)
    if interval_h <= 0 or overhead_h < 0:
        raise ValueError("interval must be > 0 and overhead >= 0")
    useful = mtbi_h * interval_h / (interval_h + checkpoint_h) - interval_h / 2
    return max(0.0, useful / (mtbi_h + overhead_h))


def _check(checkpoint_h: float, mtbi_h: float) -> None:
    if checkpoint_h <= 0 or mtbi_h <= 0:
        raise ValueError("checkpoint time and MTBI must be > 0")
