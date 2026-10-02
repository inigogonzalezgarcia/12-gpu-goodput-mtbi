# Post-incident review: goodput drop after a firmware batch (fictional)

> A practice review written against the `firmware-regression` scenario. The cluster, the firmware and the people are invented; the numbers come from the simulation (docs/results.md and `goodput analyze` on seed 1). Blameless format.

## Summary

For four weeks, the 1,024-GPU pre-training job ran at **73.5% goodput** instead of the expected 79.6% (mean of 20 simulated runs). Interruptions rose from about 45 to 64 per four weeks and the job MTBI fell from 14.0 to 9.3 hours. The cause was 32 nodes (racks 13–17) delivered with a firmware batch that made them fail about five times as often as the rest. The cost was roughly **42,000 GPU-hours** over the period: 6.2% of what the job held.

## Impact

| | Expected (baseline) | Observed (firmware-regression) |
|---|---|---|
| Goodput | 79.6% | 73.5% |
| Interruptions per 4 weeks | 45.5 | 63.6 |
| Job MTBI | 14.0 h | 9.3 h |
| Share of GPU-hours waiting for nodes | 1.2% | 6.1% |

Waiting for nodes grew five-fold: the four spares were themselves on the bad batch, so they were often in repair when a replacement was needed.

The goodput SLO (85% per 7-day window) was missed in every window, but it was missed in the baseline too, mostly because of 30-minute checkpoint intervals. **The SLO alone did not separate this incident from the job's normal state.** See action 3.

## Detection

- Week 1: the weekly goodput report showed 67% (budget burn 220%), with "waiting for nodes" as the top waste. Read as bad luck with repairs.
- Week 2: node remediation (repo 09) had drained the same nodes more than once. `goodput analyze` listed the nodes with most interruptions: four of the top five were in racks 13–17.
- Week 2: comparing node MTBF per firmware version showed batch B at about 600 hours against 3,000 hours for the rest.

## Root cause

The firmware batch on those 32 nodes raised the rate of node faults (GPU fallen off the bus, host faults). Nothing about a single interruption was unusual; only the rate per node was. The fleet had no per-firmware reliability view, so the signal had to be found by hand in the repeat-offender list.

## What went well

- Every interruption was attributed to a node and a cause, so the repeat-offender list existed.
- Remediation drained and repaired each failed node without manual work.

## What went wrong

- Spares were picked without regard to firmware version, so the reserve shared the fault.
- The SLO was already being missed for an unrelated reason, so the alert carried no news.
- No metric compared node reliability across hardware or firmware cohorts.

## Actions

1. Roll batch B back to the previous firmware, as a canary-first rollout with soak and automatic rollback (repo 11), and confirm MTBF per cohort recovers.
2. Track node MTBF per cohort (firmware, driver, hardware batch) and alert when a cohort falls below half of the fleet's.
3. Fix the known baseline waste first (checkpoint interval; see the `daly-interval` and `async-checkpoint` scenarios), so the goodput SLO is normally met and a breach means something.
4. Keep spares from more than one cohort.
5. Add a repeat-offender rule to remediation: a second incident in a short window goes to quarantine (repo 09 already does this per node; extend it to cohorts).
