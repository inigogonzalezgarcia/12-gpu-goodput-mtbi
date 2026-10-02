"""Discrete-event simulation of one large training job on a GPU node pool.

The job holds `job_nodes` nodes; the rest of the pool are spares. Interruptions arrive as a
Poisson process: each node in the job contributes 1/MTBF of its group, plus a job-level rate for
faults not tied to a node. Times are exponential, so the next interruption can be drawn again
whenever the set of nodes changes.

For each interruption the simulator writes the events of docs/event-log.md: the fault, its
detection (crash or watchdog timeout, per cause), waiting for healthy nodes if the failed node
left and no spare is free, the restart, and training resuming from the last checkpoint.

Simplifications (see docs/assumptions.md): spares and nodes in repair do not fail, faults during
detection/restart are ignored, checkpoints are blocking and of fixed duration, repair times are
exponential.
"""

from __future__ import annotations

import heapq
import itertools
import random

from goodput.events import Event, at
from goodput.scenario import Cause, Scenario


class _Node:
    __slots__ = ("name", "label", "rate")

    def __init__(self, name: str, label: str, mtbf: float):
        self.name, self.label, self.rate = name, label, 1 / mtbf


def node_inventory(s: Scenario) -> list[_Node]:
    """Pool nodes named rack-by-rack, 8 per rack: r01-n01 ... Groups fill racks in order."""
    nodes, i = [], 0
    for g in s.node_groups:
        for _ in range(g.count):
            nodes.append(_Node(f"r{i // 8 + 1:02d}-n{i % 8 + 1:02d}", g.label, g.mtbf_hours))
            i += 1
    return nodes


def simulate(s: Scenario, seed: int = 1) -> list[Event]:
    rng = random.Random(seed)
    horizon = s.days * 24
    tau = s.interval_hours()
    ckpt = s.checkpoint_minutes / 60
    restart = s.restart_minutes / 60
    pool = node_inventory(s)
    in_job: list[_Node] = pool[: s.job_nodes]
    spares: list[_Node] = pool[s.job_nodes:]
    repairs: list[tuple[float, int, _Node]] = []  # (ready time, tiebreak, node)
    job_rate = 1 / s.job_mtbi_hours
    out: list[Event] = []
    seq = itertools.count()  # deterministic tiebreak for repairs that finish at the same time

    def emit(t: float, event: str, **extra) -> None:
        out.append(Event(at(t), s.job_name, event, extra))

    def node_rate() -> float:
        return sum(n.rate for n in in_job)

    def collect_repaired(now: float) -> None:
        while repairs and repairs[0][0] <= now:
            spares.append(heapq.heappop(repairs)[2])

    emit(0, "start", nodes=s.job_nodes, gpus_per_node=s.gpus_per_node)
    t = 0.0
    while True:
        # Running: compute + checkpoints until the next interruption or the end.
        next_fail = t + rng.expovariate(node_rate() + job_rate)
        failed = False
        while True:
            ckpt_at = t + tau
            if min(next_fail, ckpt_at) >= horizon:
                emit(horizon, "end")
                return out
            if next_fail <= ckpt_at:
                t, failed = next_fail, True
                break
            t = ckpt_at
            emit(t, "checkpoint_start")
            if next_fail < t + ckpt:
                if next_fail >= horizon:
                    emit(horizon, "end")
                    return out
                t, failed = next_fail, True
                break
            if t + ckpt >= horizon:
                emit(horizon, "end")
                return out
            t += ckpt
            emit(t, "checkpoint_end")
        assert failed

        # Interruption: which node (if any) and which cause.
        nr = node_rate()
        node = None
        if rng.random() < nr / (nr + job_rate):
            node = rng.choices(in_job, weights=[n.rate for n in in_job])[0]
            cause = _pick(rng, s.node_causes)
        else:
            cause = _pick(rng, s.job_causes)
        emit(t, "interrupt", cause=cause.name, **({"node": node.name} if node else {}))

        t += cause.detect_minutes / 60
        if t >= horizon:
            emit(horizon, "end")
            return out
        emit(t, "detected")

        if node is not None and cause.node_down:
            in_job.remove(node)
            heapq.heappush(repairs, (t + rng.expovariate(1 / s.repair_hours), next(seq), node))
        collect_repaired(t)
        while len(in_job) < s.job_nodes:
            if not spares:
                ready = heapq.heappop(repairs)
                t = max(t, ready[0])
                spares.append(ready[2])
                collect_repaired(t)
            spares.sort(key=lambda n: n.name)
            in_job.append(spares.pop(0))
        if t >= horizon:
            emit(horizon, "end")
            return out
        emit(t, "nodes_ready")

        t += restart
        if t >= horizon:
            emit(horizon, "end")
            return out
        emit(t, "running")


def _pick(rng: random.Random, causes: list[Cause]) -> Cause:
    return rng.choices(causes, weights=[c.share for c in causes])[0]
