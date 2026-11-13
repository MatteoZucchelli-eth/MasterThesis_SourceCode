# LeRobot dataset preflight

This tool summarizes local LeRobot v2 datasets, checks their structural integrity, and generates a
self-contained browser dashboard. It intentionally does **not** assign a training-readiness score:
offline dataset statistics cannot guarantee closed-loop VLA performance.

## Run

```bash
dataset_tools/scripts/generate_dashboard.sh
dataset_tools/scripts/generate_dashboard.sh /path/to/data
```

Run it inside the control container (`make control`): `dq.py` needs numpy and
pyarrow, which are only installed there.

The default root is `$DATA_ROOT` or the repo's own `data/`. The script writes
the report, metrics, and dashboard under `data_analysis/result/`, then opens
the dashboard. That folder is gitignored: the artefacts are regenerated from
whatever datasets are on the machine, so they are never committed. Set
`OPEN_DASHBOARD=0` for headless use.

## What is reported

- **Overview:** episodes, frames, demonstration hours, FPS, episode durations, tasks, cameras and
  state/action dimensions.
- **Integrity:** metadata totals, episode lengths, schemas, task references, finite values,
  timestamps and expected video files.
- **Outcome:** when target-distance labels exist, their completeness, target error and the fraction
  of demonstrations that came within half the target width. An explicit per-episode success label
  is preferable when distance alone does not define success.
- **Consistency:** repeated action fraction. Descriptive only, and should be compared for the same
  task/setup.

A clean integrity result means the examined structure is internally consistent, not that a policy is
guaranteed to succeed.

## PCA target coverage

Each episode contributes one 3-D goal point. Exact target labels are used when available; otherwise
the resting point in the second half of the episode is used as a labelled proxy. Within each task,
PCA centers the points and finds the two orthogonal directions containing the most variation. This
produces the dashboard's 2-D target scatter without hard-coding the wall axes.

The 5th–95th percentile rectangle in that plane is divided by the square of the target width.
`episodes_per_target_cell` is the episode count divided by this approximate cell count. It is a
useful spatial-coverage heuristic, but has no universal pass threshold. PCA coordinates are centered
and may be rotated or mirrored; they are not literal wall coordinates.

## Deliberately excluded

Chunk-ambiguity, vision-burden, linear-R², pixel-diversity, inferred tracking-error,
frequency-domain noise, SNR and trajectory-shape scores are custom proxies without a reliable link
to VLA rollout success. The dashboard presents overview, integrity, labelled outcome, PCA coverage
and consistency instead.
