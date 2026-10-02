"""goodput: simulate GPU training-job interruptions and measure goodput and MTBI.

  goodput simulate scenarios/01-baseline.json --seed 1 -o events.jsonl
  goodput analyze events.jsonl --slo 0.85 --window-hours 168
  goodput checkpoint --mtbi-hours 14.8 --checkpoint-minutes 5
  goodput compare --seeds 20
  goodput sweep scenarios/01-baseline.json --seeds 20
  goodput report --out-dir report        # report.html + results.md
  goodput report --check docs/results.md # CI: fail if the committed results are stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from goodput import __version__, analyze, checkpoint, events, experiments, report, scenario, sim


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="goodput", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write the event log of one simulated run")
    s.add_argument("scenario")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("-o", "--output", default="-")

    a = sub.add_parser("analyze", help="goodput, MTBI and cost per cause from an event log")
    a.add_argument("log", help="JSONL event log, or - for stdin")
    a.add_argument("--slo", type=float, help="goodput SLO target, e.g. 0.85")
    a.add_argument("--window-hours", type=float, default=168)
    a.add_argument("--format", choices=("text", "json"), default="text")

    c = sub.add_parser("checkpoint", help="Young/Daly checkpoint interval and expected goodput")
    c.add_argument("--mtbi-hours", type=float, required=True)
    c.add_argument("--checkpoint-minutes", type=float, required=True)
    c.add_argument("--overhead-minutes", type=float, default=30, help="detection + restart per interruption")

    for name, helptext in (("compare", "Monte Carlo comparison of scenarios"), ("report", "HTML + Markdown report")):
        x = sub.add_parser(name, help=helptext)
        x.add_argument("--scenarios", default="scenarios")
        x.add_argument("--seeds", type=int, default=20)
        if name == "compare":
            x.add_argument("--format", choices=("md", "json"), default="md")
        else:
            x.add_argument("--out-dir", default="report")
            x.add_argument("--check", metavar="RESULTS_MD", help="only compare with a committed results file")
            x.add_argument("--slo", type=float, default=0.85)

    w = sub.add_parser("sweep", help="goodput against the checkpoint interval")
    w.add_argument("scenario")
    w.add_argument("--seeds", type=int, default=20)
    w.add_argument("--intervals", help="comma-separated minutes")

    args = p.parse_args(argv)
    try:
        return COMMANDS[args.cmd](args)
    except (ValueError, OSError) as e:
        print(f"goodput: {e}", file=sys.stderr)
        return 2


def cmd_simulate(args) -> int:
    evs = sim.simulate(scenario.load(args.scenario), args.seed)
    if args.output == "-":
        events.write(evs, sys.stdout)
    else:
        with open(args.output, "w") as f:
            events.write(evs, f)
    return 0


def cmd_analyze(args) -> int:
    src = sys.stdin if args.log == "-" else open(args.log)
    with src:
        jobs = analyze.build_jobs(events.read(src))
    if not jobs:
        raise ValueError("no events in the log")
    s = analyze.summarize(jobs)
    slo_res = analyze.slo(jobs, args.slo, args.window_hours) if args.slo else None
    print(report.summary_json(s, slo_res) if args.format == "json" else report.summary_text(s, slo_res))
    return 0


def cmd_checkpoint(args) -> int:
    m, c = args.mtbi_hours, args.checkpoint_minutes / 60
    y, d = checkpoint.young(c, m), checkpoint.daly(c, m)
    print(f"MTBI {m:g} h, checkpoint write {args.checkpoint_minutes:g} min")
    print(f"Young: {y * 60:.1f} min   Daly: {d * 60:.1f} min")
    print("interval  expected goodput (first-order)")
    for minutes in sorted({5, 10, 15, 30, 60, 120, 240, round(d * 60)}):
        g = checkpoint.expected_goodput(m, c, minutes / 60, args.overhead_minutes / 60)
        print(f"{minutes:>6} min  {report.pct(g)}{'   <- Daly' if minutes == round(d * 60) else ''}")
    return 0


def _scenarios(path: str):
    found = scenario.load_dir(path)
    if not found:
        raise ValueError(f"no scenario files in {path}")
    return found


def cmd_compare(args) -> int:
    results = experiments.compare(_scenarios(args.scenarios), args.seeds)
    if args.format == "json":
        print(json.dumps([{"scenario": r.scenario.name, "goodput_mean": round(r.mean, 5), "p10": round(r.p10, 5),
                           "p90": round(r.p90, 5), "interruptions": round(r.interruptions, 2),
                           "mtbi_hours": round(r.mtbi_hours, 2),
                           "shares": {k: round(v, 5) for k, v in r.shares.items()}} for r in results], indent=2))
    else:
        print("| Scenario | Goodput | p10–p90 | Interruptions | MTBI |\n|---|---|---|---|---|")
        for r in results:
            print(f"| {r.scenario.name} | {report.pct(r.mean)} | {report.pct(r.p10)}–{report.pct(r.p90)} | "
                  f"{r.interruptions:.1f} | {r.mtbi_hours:.1f} h |")
    return 0


def cmd_sweep(args) -> int:
    s = scenario.load(args.scenario)
    intervals = [float(x) for x in args.intervals.split(",")] if args.intervals else experiments.DEFAULT_INTERVALS
    sw = experiments.sweep(s, args.seeds, intervals)
    print(f"Young {sw.young_minutes:.0f} min, Daly {sw.daly_minutes:.0f} min (simulated {report.pct(sw.daly_simulated)})")
    for pt in sw.points:
        print(f"{pt.interval_minutes:>6g} min  simulated {report.pct(pt.simulated)}  model {report.pct(pt.model)}")
    return 0


def cmd_report(args) -> int:
    scenarios = _scenarios(args.scenarios)
    results = experiments.compare(scenarios, args.seeds)
    sw = experiments.sweep(scenarios[0], args.seeds)
    md = report.results_markdown(results, sw)
    if args.check:
        committed = Path(args.check).read_text()
        if committed != md:
            print(f"{args.check} is out of date: run `goodput report --out-dir report` and copy report/results.md",
                  file=sys.stderr)
            return 1
        print(f"{args.check} matches the simulation")
        return 0
    jobs = analyze.build_jobs(sim.simulate(scenarios[0], 1))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(md)
    (out / "report.html").write_text(report.results_html(results, sw, analyze.summarize(jobs),
                                                         analyze.slo(jobs, args.slo, 168)))
    print(f"wrote {out / 'report.html'} and {out / 'results.md'}")
    return 0


COMMANDS = {"simulate": cmd_simulate, "analyze": cmd_analyze, "checkpoint": cmd_checkpoint,
            "compare": cmd_compare, "sweep": cmd_sweep, "report": cmd_report}
