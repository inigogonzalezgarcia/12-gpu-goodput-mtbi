"""Turn an event log into time spans, then into goodput, MTBI and where the time went.

Every second of a job's wall-clock time falls in exactly one category:

    productive   compute whose result was kept (saved by a later checkpoint, or the job ended)
    checkpoint   writing checkpoints (blocking)
    lost         compute done after the last checkpoint and thrown away by an interruption
    detect       the job is dead or hung but nobody knows yet
    wait         waiting for healthy nodes to replace the failed ones
    restart      rescheduling, loading the checkpoint, warming up

goodput = productive / total. Lost, detect, wait and restart are charged to the interruption that
caused them, so each cause gets a cost in GPU-hours.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from goodput.events import Event

CATEGORIES = ("productive", "checkpoint", "lost", "detect", "wait", "restart")
WASTE = ("checkpoint", "lost", "detect", "wait", "restart")


@dataclass
class Span:
    start: datetime
    end: datetime
    category: str
    cause: str = ""  # interruption the span is charged to (lost, detect, wait, restart)

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600


@dataclass
class Interruption:
    time: datetime
    cause: str
    node: str | None


@dataclass
class Job:
    name: str
    nodes: int
    gpus_per_node: int
    start: datetime
    end: datetime
    spans: list[Span] = field(default_factory=list)
    interruptions: list[Interruption] = field(default_factory=list)

    @property
    def gpus(self) -> int:
        return self.nodes * self.gpus_per_node


def build_jobs(events: list[Event]) -> list[Job]:
    by_job: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_job[e.job].append(e)
    return [_build(name, evs) for name, evs in sorted(by_job.items())]


# state -> category of the span that ends at the next event
_STATE_CAT = {"compute": "compute", "checkpoint": "checkpoint", "detect": "detect", "wait": "wait", "restart": "restart"}
# event -> states it may follow
_ALLOWED = {
    "checkpoint_start": {"compute"},
    "checkpoint_end": {"checkpoint"},
    "interrupt": {"compute", "checkpoint"},
    "detected": {"detect"},
    "nodes_ready": {"wait"},
    "running": {"restart"},
    "end": {"compute", "checkpoint", "detect", "wait", "restart"},
}
_NEXT = {"checkpoint_start": "checkpoint", "checkpoint_end": "compute", "interrupt": "detect",
         "detected": "wait", "nodes_ready": "restart", "running": "compute"}


def _build(name: str, evs: list[Event]) -> Job:
    if evs[0].event != "start":
        raise ValueError(f"job {name}: first event must be start, got {evs[0].event}")
    if evs[-1].event != "end":
        raise ValueError(f"job {name}: last event must be end, got {evs[-1].event}")
    s = evs[0]
    job = Job(name, int(s.extra.get("nodes", 1)), int(s.extra.get("gpus_per_node", 8)), s.time, evs[-1].time)
    state, last, cause = "compute", s.time, ""
    uncommitted: list[Span] = []  # compute not yet saved by a checkpoint

    for e in evs[1:]:
        if e.event == "start":
            raise ValueError(f"job {name}: second start at {e.time}")
        if state not in _ALLOWED[e.event]:
            raise ValueError(f"job {name}: {e.event} at {e.time} not allowed while {state}")
        cat = _STATE_CAT[state]
        span = Span(last, e.time, cat, cause if cat in WASTE[2:] else "")
        if span.end > span.start:
            job.spans.append(span)
            if cat == "compute":
                uncommitted.append(span)
        last = e.time

        if e.event == "checkpoint_end":
            for sp in uncommitted:
                sp.category = "productive"
            uncommitted = []
        elif e.event == "interrupt":
            cause = str(e.extra.get("cause", "unknown"))
            job.interruptions.append(Interruption(e.time, cause, e.extra.get("node")))
            for sp in uncommitted:
                sp.category, sp.cause = "lost", cause
            uncommitted = []
        elif e.event == "end":
            # Work in flight at the end of the log is counted as kept: the job would save it next.
            for sp in uncommitted:
                sp.category = "productive"
            break
        if e.event == "running":
            cause = ""
        state = _NEXT.get(e.event, state)
    return job


@dataclass
class Summary:
    jobs: int
    gpus: int
    wall_hours: float
    gpu_hours: dict[str, float]
    interruptions: int
    running_hours: float
    node_running_hours: float
    node_interruptions: int
    by_cause: dict[str, dict]
    nodes: Counter

    @property
    def total_gpu_hours(self) -> float:
        return sum(self.gpu_hours.values())

    @property
    def goodput(self) -> float:
        t = self.total_gpu_hours
        return self.gpu_hours["productive"] / t if t else 0.0

    @property
    def mtbi_hours(self) -> float | None:
        """Running time (compute + checkpointing) between interruptions."""
        return self.running_hours / self.interruptions if self.interruptions else None

    @property
    def node_mtbf_hours(self) -> float | None:
        """Node-hours in a running job per node-attributed interruption."""
        return self.node_running_hours / self.node_interruptions if self.node_interruptions else None


def summarize(jobs: list[Job], start: datetime | None = None, end: datetime | None = None) -> Summary:
    """Aggregate jobs, optionally only the part of each span inside [start, end)."""
    gh = {c: 0.0 for c in CATEGORIES}
    by_cause: dict[str, dict] = defaultdict(lambda: {"count": 0, "gpu_hours": 0.0})
    running = node_running = 0.0
    interruptions = node_int = 0
    nodes: Counter = Counter()
    wall = 0.0
    for j in jobs:
        lo, hi = max(j.start, start or j.start), min(j.end, end or j.end)
        if hi <= lo:
            continue
        wall = max(wall, (hi - lo).total_seconds() / 3600)
        for sp in j.spans:
            h = _overlap(sp, lo, hi)
            if h <= 0:
                continue
            gh[sp.category] += h * j.gpus
            if sp.category in ("productive", "checkpoint", "lost"):
                running += h
                node_running += h * j.nodes
            if sp.cause:
                by_cause[sp.cause]["gpu_hours"] += h * j.gpus
        for i in j.interruptions:
            if lo <= i.time < hi:
                interruptions += 1
                by_cause[i.cause]["count"] += 1
                if i.node:
                    node_int += 1
                    nodes[i.node] += 1
    return Summary(len(jobs), sum(j.gpus for j in jobs), wall, gh, interruptions, running, node_running,
                   node_int, dict(by_cause), nodes)


def windows(jobs: list[Job], hours: float) -> list[tuple[datetime, datetime, Summary]]:
    if not jobs:
        return []
    start, end = min(j.start for j in jobs), max(j.end for j in jobs)
    step = timedelta(hours=hours)
    out, t = [], start
    while t < end:
        out.append((t, min(t + step, end), summarize(jobs, t, min(t + step, end))))
        t += step
    return out


@dataclass
class SLOResult:
    target: float
    window_hours: float
    rows: list[dict]

    @property
    def met(self) -> int:
        return sum(1 for r in self.rows if r["goodput"] >= self.target)


def slo(jobs: list[Job], target: float, window_hours: float) -> SLOResult:
    """Goodput SLO per window. The error budget is the share of GPU-hours allowed to be
    unproductive, (1 - target) * total; burn is how much of it the window used."""
    if not 0 < target < 1:
        raise ValueError("SLO target must be between 0 and 1")
    rows = []
    for lo, hi, s in windows(jobs, window_hours):
        total = s.total_gpu_hours
        budget = (1 - target) * total
        spent = total - s.gpu_hours["productive"]
        rows.append({"start": lo, "end": hi, "goodput": s.goodput, "interruptions": s.interruptions,
                     "budget_burn": spent / budget if budget else 0.0,
                     "top_waste": max(WASTE, key=lambda c: s.gpu_hours[c]) if spent > 0 else "-"})
    return SLOResult(target, window_hours, rows)


def _overlap(sp: Span, lo: datetime, hi: datetime) -> float:
    a, b = max(sp.start, lo), min(sp.end, hi)
    return (b - a).total_seconds() / 3600 if b > a else 0.0
