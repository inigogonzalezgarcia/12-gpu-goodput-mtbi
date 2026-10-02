# Event log format

`goodput analyze` reads JSON Lines: one event per line, any number of jobs, sorted by time when read. Lines starting with `#` are ignored. Times must carry a timezone (`Z` or an offset).

| Event | Meaning | Extra fields |
|---|---|---|
| `start` | The job starts training | `nodes`, `gpus_per_node` |
| `checkpoint_start` | Training blocks to write a checkpoint | |
| `checkpoint_end` | The checkpoint is durable; everything computed before it is kept | |
| `interrupt` | The fault happens | `cause`, optionally `node` |
| `detected` | The job is known to be dead (crash noticed, or watchdog fired) | |
| `nodes_ready` | Enough healthy nodes are allocated again | |
| `running` | Training resumes from the last checkpoint | |
| `end` | End of the job or of the log | |

Allowed order per job: `start`, then any number of `checkpoint_start → checkpoint_end` while running; an `interrupt` (while computing or checkpointing) must be followed by `detected → nodes_ready → running`; `end` may come at any point. Anything else is rejected with the line and event that broke the sequence.

[examples/sample-events.jsonl](../examples/sample-events.jsonl) is a hand-written log with every event; the unit tests check its numbers by hand.

## Building it from a real cluster

Not tested in this lab: it is a description of where each event would come from.

| Event | Possible source |
|---|---|
| `start`, `end` | Scheduler accounting (Slurm `sacct` start/end, Kubernetes Job or JobSet status) |
| `checkpoint_start`, `checkpoint_end` | Training framework logs (checkpoint save begin/finish) |
| `interrupt` | First fault signal: XID in the kernel log or DCGM, NCCL error, process exit. `node` from the host that reported it |
| `detected` | Job controller or watchdog marking the job failed |
| `nodes_ready` | Scheduler allocation of the replacement node (Slurm requeue start, pod scheduled) |
| `running` | First training step logged after the restart |

The hard part in practice is the cause: the first signal is often a symptom on a different node. Repo 09 (node remediation) is one way to attach a cause to a node.
