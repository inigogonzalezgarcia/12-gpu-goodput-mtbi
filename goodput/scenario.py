"""Scenario files: the cluster, the job and the failure model the simulator runs.

All numbers are lab assumptions, chosen to sit between published reference points (see
docs/assumptions.md). They are not measurements of any real cluster.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from goodput.checkpoint import daly


@dataclass
class Cause:
    name: str
    share: float  # share of the interruptions of its kind (node-level or job-level)
    detect_minutes: float  # fault -> job known to be dead
    node_down: bool  # the node leaves the job and goes to repair
    description: str = ""


@dataclass
class NodeGroup:
    count: int
    mtbf_hours: float  # node-attributable interruptions only
    label: str = "default"


@dataclass
class Scenario:
    name: str
    description: str
    job_nodes: int
    gpus_per_node: int
    node_groups: list[NodeGroup]
    job_mtbi_hours: float  # interruptions not tied to a node (software, storage, framework)
    node_causes: list[Cause]
    job_causes: list[Cause]
    checkpoint_minutes: float
    checkpoint_interval_minutes: float | str  # minutes, or "daly"
    restart_minutes: float
    repair_hours: float
    days: float = 28
    job_name: str = "llm-pretrain"
    extra: dict = field(default_factory=dict)

    @property
    def pool_nodes(self) -> int:
        return sum(g.count for g in self.node_groups)

    @property
    def spares(self) -> int:
        return self.pool_nodes - self.job_nodes

    def nominal_mtbi_hours(self) -> float:
        """Expected job MTBI if the job ran on nodes picked evenly from the pool."""
        per_node = sum(g.count / g.mtbf_hours for g in self.node_groups) / self.pool_nodes
        return 1 / (self.job_nodes * per_node + 1 / self.job_mtbi_hours)

    def interval_hours(self) -> float:
        if self.checkpoint_interval_minutes == "daly":
            return daly(self.checkpoint_minutes / 60, self.nominal_mtbi_hours())
        return float(self.checkpoint_interval_minutes) / 60

    def mean_detect_hours(self) -> float:
        """Average detection time, weighted by how often each cause happens."""
        node_rate = 1 / self.nominal_mtbi_hours() - 1 / self.job_mtbi_hours
        job_rate = 1 / self.job_mtbi_hours
        d = sum(c.share * c.detect_minutes for c in self.node_causes) * node_rate
        d += sum(c.share * c.detect_minutes for c in self.job_causes) * job_rate
        return d / (node_rate + job_rate) / 60

    def validate(self) -> None:
        errs = []
        if self.job_nodes < 1 or self.gpus_per_node < 1:
            errs.append("job_nodes and gpus_per_node must be >= 1")
        if self.pool_nodes < self.job_nodes:
            errs.append(f"node_groups have {self.pool_nodes} nodes, the job needs {self.job_nodes}")
        for kind, causes in (("node_causes", self.node_causes), ("job_causes", self.job_causes)):
            if causes and abs(sum(c.share for c in causes) - 1) > 1e-6:
                errs.append(f"{kind} shares must add up to 1")
        if any(g.mtbf_hours <= 0 for g in self.node_groups) or self.job_mtbi_hours <= 0:
            errs.append("MTBF/MTBI values must be > 0")
        if self.checkpoint_minutes <= 0 or self.restart_minutes < 0 or self.repair_hours <= 0 or self.days <= 0:
            errs.append("checkpoint_minutes, repair_hours and days must be > 0, restart_minutes >= 0")
        if self.checkpoint_interval_minutes != "daly":
            try:
                if float(self.checkpoint_interval_minutes) <= 0:
                    errs.append("checkpoint_interval_minutes must be > 0 or \"daly\"")
            except (TypeError, ValueError):
                errs.append("checkpoint_interval_minutes must be a number or \"daly\"")
        if errs:
            raise ValueError(f"scenario {self.name}: " + "; ".join(errs))


_FIELDS = {"name", "description", "job_nodes", "gpus_per_node", "node_groups", "job_mtbi_hours", "node_causes",
           "job_causes", "checkpoint_minutes", "checkpoint_interval_minutes", "restart_minutes", "repair_hours",
           "days", "job_name", "base"}


def load(path: str | Path, _seen: tuple = ()) -> Scenario:
    """Load a scenario. A scenario may name a `base` file and override only some fields."""
    path = Path(path)
    if path in _seen:
        raise ValueError(f"scenario {path}: base loop")
    raw = json.loads(path.read_text())
    unknown = set(raw) - _FIELDS
    if unknown:
        raise ValueError(f"scenario {path}: unknown field(s) {sorted(unknown)}")
    if "base" in raw:
        base = _to_dict(load(path.parent / raw.pop("base"), _seen + (path,)))
        base.update(raw)
        raw = base
    s = Scenario(
        name=raw["name"], description=raw.get("description", ""),
        job_nodes=raw["job_nodes"], gpus_per_node=raw.get("gpus_per_node", 8),
        node_groups=[NodeGroup(**g) for g in raw["node_groups"]],
        job_mtbi_hours=raw["job_mtbi_hours"],
        node_causes=[Cause(**c) for c in raw["node_causes"]],
        job_causes=[Cause(**c) for c in raw["job_causes"]],
        checkpoint_minutes=raw["checkpoint_minutes"],
        checkpoint_interval_minutes=raw["checkpoint_interval_minutes"],
        restart_minutes=raw["restart_minutes"], repair_hours=raw["repair_hours"],
        days=raw.get("days", 28), job_name=raw.get("job_name", "llm-pretrain"),
    )
    s.validate()
    return s


def _to_dict(s: Scenario) -> dict:
    return {
        "name": s.name, "description": s.description, "job_nodes": s.job_nodes, "gpus_per_node": s.gpus_per_node,
        "node_groups": [vars(g).copy() for g in s.node_groups], "job_mtbi_hours": s.job_mtbi_hours,
        "node_causes": [vars(c).copy() for c in s.node_causes], "job_causes": [vars(c).copy() for c in s.job_causes],
        "checkpoint_minutes": s.checkpoint_minutes, "checkpoint_interval_minutes": s.checkpoint_interval_minutes,
        "restart_minutes": s.restart_minutes, "repair_hours": s.repair_hours, "days": s.days, "job_name": s.job_name,
    }


def load_dir(path: str | Path) -> list[Scenario]:
    return [load(p) for p in sorted(Path(path).glob("*.json"))]
