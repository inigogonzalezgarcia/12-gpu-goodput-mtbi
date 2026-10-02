# Definitions

## Where the time goes

The analyzer puts every second of a job's wall-clock time into one category and multiplies by the job's GPUs, so the categories always add up to the GPU-hours the job held.

| Category | What it is |
|---|---|
| Productive | Compute whose result was kept: saved by a later checkpoint, or still in flight when the log ends |
| Checkpoint writes | Time training is blocked writing a checkpoint |
| Lost work | Compute done after the last checkpoint and thrown away by an interruption |
| Detection | From the fault to the moment the job is known to be dead. A crash is seen at once; a hang only when a watchdog times out |
| Waiting for nodes | The failed node left and no healthy spare was free |
| Restart | Rescheduling, loading the checkpoint, warming up |

Lost work, detection, waiting and restart are charged to the interruption that caused them, which gives each cause a cost in GPU-hours. Counting interruptions alone hides that a 30-minute hang costs far more than a crash seen in seconds.

## Goodput

**goodput = productive GPU-hours / GPU-hours held by the job**

This is the same idea as the Effective Training Time Ratio (ETTR) in Meta's *Revisiting Reliability in Large-Scale Machine Learning Research Clusters* (2024): productive runtime over available wall-clock time, where checkpoint writes, restarts and recomputing lost work are not productive. The paper also counts time a job waits in the queue; this lab models one job that holds its nodes, so queueing only appears as "waiting for nodes" after a failure.

Goodput here is about the infrastructure, not the model: a job that runs with poor GPU utilisation still counts as productive.

## MTBI and node MTBF

- **MTBI (mean time between interruptions)** of a job = running time (compute + checkpoint writes) / interruptions. Time spent detecting or restarting is excluded, because a job cannot be interrupted while it is already down.
- **Node MTBF** = node-hours spent in a running job / interruptions attributed to a node.

They are linked by the job size. With N nodes, each failing on average every MTBF hours, and a rate of job-level faults (software, storage) every J hours:

    1 / MTBI = N / MTBF + 1 / J

So a node that fails once every 3,000 hours looks very reliable, but 128 of them interrupt a job every 23 hours, and with software faults added, every 15 hours. Doubling the job halves the MTBI. This is why large training jobs care about MTBI and fleet operators about node MTBF, and why reports should give both.

## Checkpoint interval

With checkpoint write time C and job MTBI M:

- **Young (1974):** τ = √(2CM)
- **Daly (2006):** τ = √(2CM) · [1 + ⅓·√(C/2M) + ⅑·(C/2M)] − C, for C < 2M; τ = M otherwise

τ is compute time between checkpoints. Daly shows the restart time does not move the optimum. Checkpointing too often wastes time writing; too rarely, each interruption throws away more work. The curve is flat near the optimum, so being within ±30% of it costs little.

The **first-order model** (`goodput checkpoint`) estimates goodput as

    goodput ≈ (M · τ/(τ+C) − τ/2) / (M + overhead per interruption)

It ignores waiting for nodes, which is why it sits about one point above the simulation in the baseline (see [results.md](results.md)).

## SLO and error budget

A goodput SLO of 85% over 7-day windows allows 15% of the GPU-hours in each window to be unproductive. That 15% is the error budget. Budget burn is the unproductive share divided by the budget: 100% means the window used it all. Because every category is measured, a window that misses its SLO also shows *what* used the budget (checkpoint writes, waiting for nodes…), which is where the next improvement should go.
