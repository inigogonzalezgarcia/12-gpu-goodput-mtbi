# Assumptions

Everything in `scenarios/` is a lab assumption, not a measurement. The aim is numbers that are plausible enough to make the trade-offs visible, and documented well enough that anyone can change them.

## Reference points

Two published figures for large training jobs:

- Meta's Llama 3 paper (*The Llama 3 Herd of Models*, 2024) reports 419 unexpected interruptions in a 54-day period of pre-training on 16,384 H100 GPUs: about one every 3 hours for the whole job. Scaled linearly to 1,024 GPUs, that is about one every 50 hours.
- Meta's *Revisiting Reliability in Large-Scale Machine Learning Research Clusters* (2024) reports a mean time to failure of about 7.9 hours for 1,024-GPU jobs on its research clusters.

The baseline job (1,024 GPUs) has a nominal MTBI of 14.8 hours, between the two.

## Baseline parameters

| Parameter | Value | Why |
|---|---|---|
| Job | 128 nodes × 8 GPUs | A common building block for pre-training |
| Spares | 4 nodes | A small reserve; see the `no-spares` scenario for none |
| Node MTBF | 3,000 h | Node-attributable interruptions only |
| Job-level MTBI | 40 h | Software crashes and storage stalls, not tied to a node |
| Checkpoint write | 5 min, blocking | Synchronous save of a large model to shared storage |
| Checkpoint interval | 30 min | A typical fixed setting, deliberately not optimal |
| Hang detection | 30 min | A conservative watchdog timeout |
| Crash detection | 1–5 min | The process exits; the job controller notices quickly |
| Restart | 20 min | Reschedule, load the checkpoint, warm up |
| Repair | 48 h mean (exponential) | Drain, diagnose, reset or replace, validate (repos 09 and 11) |

Causes of node interruptions and their shares: GPU fallen off the bus (XID 79) 25%, GPU memory (uncorrectable ECC, row remapping) 20%, NVLink/InfiniBand link 15%, host 10%, collective hang with no hardware fault 30%. Job-level: software crash 60%, storage stall 40%. Only node faults with hardware behind them send the node to repair.

## Simplifications

- One job that holds its nodes for the whole period. No queue, no other jobs competing for spares.
- Interruptions follow a Poisson process (exponential times). Real fleets have infant mortality on new hardware and bursts after changes, which this does not model.
- Spare nodes and nodes in repair do not fail. Faults during detection or restart are ignored.
- Checkpoints take a fixed time. Asynchronous checkpointing is modelled as a short blocking snapshot.
- A node back from repair is as good as new.
- Work in flight at the end of the simulated period counts as productive.

## Determinism

Runs are seeded (seeds 1…N), so every number in [results.md](results.md) can be reproduced, and CI checks that it still is. All scenarios use the same seeds, so they see similar interruption sequences and their differences are mostly the effect of the change, not luck.
