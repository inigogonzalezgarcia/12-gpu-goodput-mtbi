# GPU Goodput and MTBI

How much of a large GPU training job's time actually trains the model, where the rest goes, and which change gets it back. A simulator of interruptions on a GPU node pool, an analyzer that turns an event log into goodput, MTBI and GPU-hours lost per cause, the Young/Daly checkpoint formulas, and a goodput SLO with an error budget.

**Interruptions → lost work, detection, waiting, restart → goodput, MTBI, cost per cause → what to fix first**

![ci](https://github.com/inigogonzalezgarcia/12-gpu-goodput-mtbi/actions/workflows/ci.yml/badge.svg)

> A learning-in-public lab about reliability for large GPU clusters. I don't have production GPU fleet experience; this project is how I am learning the problem. All numbers come from a simulation with assumptions documented in [docs/assumptions.md](docs/assumptions.md), set between two published reference points (Meta's Llama 3 report and Meta's research-cluster reliability paper). They are not measurements of any real cluster.

Fourth in a series: [09 – node remediation](https://github.com/inigogonzalezgarcia/09-gpu-node-remediation), [10 – fleet observability](https://github.com/inigogonzalezgarcia/10-gpu-fleet-observability), [11 – fleet lifecycle](https://github.com/inigogonzalezgarcia/11-gpu-fleet-lifecycle). Those repos keep nodes healthy; this one measures what that is worth to the jobs.

## What the simulation shows

A 1,024-GPU job (128 nodes × 8 GPUs, 4 spares) over 28 days, 20 seeds per scenario:

| Scenario | Goodput | vs baseline | Biggest remaining waste |
|---|---|---|---|
| baseline: checkpoint every 30 min (5 min to write), 30-min hang watchdog | 79.6% | – | Checkpoint writes (13.3%) |
| checkpoint interval from Daly's formula (91 min) | 85.1% | +5.4 pp | Lost work (5.2%) |
| asynchronous checkpoints (30 s blocking) | 91.7% | +12.0 pp | Restart (2.3%) |
| hangs detected in 5 min instead of 30 | 80.6% | +0.9 pp | Checkpoint writes (13.5%) |
| no spare nodes | 36.1% | −43.5 pp | Waiting for nodes (55.4%) |
| 32 nodes on a bad firmware batch | 73.5% | −6.2 pp | Checkpoint writes (12.3%) |
| combined: async + Daly + fast detection + 8 spares | 94.0% | +14.4 pp | Restart (2.3%) |

Full tables with p10–p90 ranges, the time breakdown and the checkpoint sweep: [docs/results.md](docs/results.md). CI regenerates that file and fails if it no longer matches the code.

What I take from it:

- **Measure where the time goes before tuning anything.** In the baseline, checkpoint writes cost about four times as much as every hang-type interruption put together (lost work, detection and restart included). Faster hang detection is worth doing, but it is not the first fix.
- **Spare capacity matters more than any other single setting.** Without spares every hardware fault waits for a repair.
- **The checkpoint interval has a cheap, well-known optimum.** Daly's formula landed as well as the best interval found by brute force.
- **Rates hide in averages.** The firmware regression looks like ordinary failures one at a time. It only shows up as a cohort with a lower node MTBF ([post-incident review](docs/post-incident-review.md)).

## Run it

Python 3.11+, standard library only.

```bash
python -m goodput checkpoint --mtbi-hours 14.8 --checkpoint-minutes 5      # Young/Daly + expected goodput
python -m goodput simulate scenarios/01-baseline.json --seed 1 -o events.jsonl
python -m goodput analyze events.jsonl --slo 0.85 --window-hours 168       # goodput, MTBI, cost per cause, SLO
python -m goodput compare --seeds 20                                       # all scenarios
python -m goodput sweep scenarios/01-baseline.json                         # goodput vs checkpoint interval
python -m goodput report --out-dir report                                  # report.html + results.md
python -m unittest -v
```

`goodput analyze` on one baseline run:

```
Goodput: 78.5%
Interruptions: 44  MTBI: 14.3 h  node MTBF (estimated): 2682.4 h

Cost per interruption cause (lost work + detection + waiting + restart):
  gpu_xid79             9 x      23,328 GPU-h
  collective_hang       9 x      10,734 GPU-h
  storage_stall         8 x       9,050 GPU-h
  ...
Goodput SLO 85% per 168 h window: met in 0 of 4
  2026-09-01  73.3%  budget burn  178%    9 interruptions  top waste: checkpoint
```

CI also builds the HTML report (one self-contained file with charts) and attaches it to each run as the `goodput-report` artifact.

## How it works

| Piece | What it does |
|---|---|
| `goodput/sim.py` | Discrete-event simulation: Poisson interruptions per node and per job, cause-specific detection times, spares and repairs, blocking checkpoints. Writes an event log |
| `goodput/analyze.py` | Reads any event log in the [documented format](docs/event-log.md), puts every second into one category, charges waste to causes, computes MTBI, node MTBF and SLO windows |
| `goodput/checkpoint.py` | Young and Daly intervals and a first-order goodput model |
| `goodput/experiments.py` | Monte Carlo over seeds: scenario comparison and interval sweep |
| `scenarios/*.json` | The cluster, job and failure model; each scenario changes one thing from the baseline |

## Documentation

- [docs/definitions.md](docs/definitions.md): goodput, MTBI vs node MTBF, the checkpoint formulas, SLO and error budget
- [docs/assumptions.md](docs/assumptions.md): every parameter, the reference points and the simplifications
- [docs/event-log.md](docs/event-log.md): the log format and where each event would come from on a real cluster
- [docs/post-incident-review.md](docs/post-incident-review.md): a fictional review of the firmware regression
- [docs/decisions.md](docs/decisions.md): design decisions

## Roadmap

- Feed the analyzer from a Slurm cluster's accounting and a training framework's logs (next repo in the series).
- Non-exponential failures: infant mortality on new hardware and bursts after changes.
- Several jobs competing for the same spares.
- Elastic training that continues on fewer nodes instead of waiting.

## Customisation and contact

Want to talk about GPU cluster reliability, goodput or a lab like this for your team? Get in touch:

- Email: [inigogonzalezgarcia@yahoo.es](mailto:inigogonzalezgarcia@yahoo.es)
- LinkedIn: [linkedin.com/in/igonzalez93](https://www.linkedin.com/in/igonzalez93)

## License

MIT
