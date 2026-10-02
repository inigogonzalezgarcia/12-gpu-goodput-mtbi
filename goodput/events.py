"""The event log: one JSON object per line, one stream per training job.

    {"time": "2026-09-01T00:00:00Z", "job": "llm-pretrain", "event": "start", "nodes": 128, "gpus_per_node": 8}
    {"time": "...", "job": "llm-pretrain", "event": "checkpoint_start"}
    {"time": "...", "job": "llm-pretrain", "event": "checkpoint_end"}
    {"time": "...", "job": "llm-pretrain", "event": "interrupt", "cause": "gpu_xid79", "node": "r03-n05"}
    {"time": "...", "job": "llm-pretrain", "event": "detected"}
    {"time": "...", "job": "llm-pretrain", "event": "nodes_ready"}
    {"time": "...", "job": "llm-pretrain", "event": "running"}
    {"time": "...", "job": "llm-pretrain", "event": "end"}

`interrupt` is when the fault happens, `detected` when the job is known to be dead (a crash is
seen at once, a hang only when a watchdog times out), `nodes_ready` when enough healthy nodes are
allocated again, `running` when training resumes from the last checkpoint. The simulator writes
this format; a real cluster can produce it from scheduler and training-framework logs (see
docs/event-log.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import IO, Iterable

EVENTS = ("start", "checkpoint_start", "checkpoint_end", "interrupt", "detected", "nodes_ready", "running", "end")
EPOCH = datetime(2026, 9, 1, tzinfo=timezone.utc)


@dataclass
class Event:
    time: datetime
    job: str
    event: str
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = {"time": self.time.strftime("%Y-%m-%dT%H:%M:%SZ"), "job": self.job, "event": self.event}
        d.update(self.extra)
        return json.dumps(d)


def at(hours: float) -> datetime:
    """Simulation hours since the lab epoch, rounded to the second."""
    return EPOCH + timedelta(seconds=round(hours * 3600))


def parse_time(s: str) -> datetime:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"timestamp without a timezone: {s}")
    return t


def read(lines: Iterable[str]) -> list[Event]:
    out = []
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            d = json.loads(line)
            ev = d.pop("event")
            if ev not in EVENTS:
                raise ValueError(f"unknown event {ev!r}")
            out.append(Event(parse_time(d.pop("time")), str(d.pop("job")), ev, d))
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"line {n}: {e}") from None
    out.sort(key=lambda e: e.time)  # stable: events with the same timestamp keep file order
    return out


def write(events: Iterable[Event], f: IO[str]) -> None:
    for e in events:
        f.write(e.to_json() + "\n")
