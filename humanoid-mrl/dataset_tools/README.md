# dataset_tools — operating on recorded LeRobot datasets

Everything here acts on a dataset that already exists on disk. Recording and
collection live elsewhere (`scripts/collect_workers.sh`,
`src/g1_teleoperation/scripts/start_recording.bash`); the aggregate
quality/dashboard analysis is `data_analysis/`.

Layout: the entry points you run are the bash wrappers in `dataset_tools/scripts/`; the
Python they drive stays at the top level, so the `python -m dataset_tools.*`
module paths keep working. Run everything **from the repo root** —
`dataset_tools` is a namespace package and those module paths only resolve
from there.

| Entry point | What it does |
|---|---|
| `dataset_tools/scripts/sync_to_euler.sh <dataset>` | Push a dataset to Euler for finetuning, into `/cluster/work/mrl/$USER/datasets/<name>/` |
| `dataset_tools/scripts/merge_datasets.sh --out <root> <worker-root>...` | Merge per-worker collection roots into one globally re-indexed dataset |
| `dataset_tools/scripts/data_analysis.sh <dataset> [out-dir]` | Run per-episode trajectory, target-overlay, and synchronization checks |
| `dataset_tools/scripts/generate_dashboard.sh [data-root]` | Generate aggregate metrics, report, and dashboard |
| `dataset_tools/scripts/clean_dataset.sh <dataset>` | Delete slip and black-camera episodes, then reindex |
| `dataset_tools/scripts/generate_pixel_labels.sh <dataset>` | Generate per-frame wrist-camera target labels |
| `python dataset_tools/delete_episode.py --episode N...` | Delete episodes by index and reindex the rest |
| `python dataset_tools/visualize_episode.py` | Plot one episode's state/action/reward/tip-target panels |
| `python dataset_tools/visualize_episodes.py --dataset <d>` | The same panels over a random sample of episodes |
| `python -m dataset_tools.heatmap_labels.<mod>` | Generate and check the projected target labels (`process_dataset`, `overlay_check`, `sync_check`) |

## Which environment each one needs

Run all Python tools inside the control container. Its environment provides
the dependencies from the repository `requirements.txt`; no extra image or
virtual environment is needed.

```bash
make control
docker compose -f docker/docker-compose.yaml exec dev-container bash
dataset_tools/scripts/clean_dataset.sh data/small_targets/<dataset> --dry-run
```

The wrappers activate the container environment and run from the repo root.

The bare `python dataset_tools/*.py` and `python -m dataset_tools.heatmap_labels.*`
entry points have no such guard — run them inside the container too, from the
repo root.

`sync_to_euler.sh` runs on the host because it only needs ssh and rsync. It
uses the `euler` ssh alias by default; override it with `EULER_HOST`.

Files written from inside the container are owned by root on the host;
`scripts/fix_ownership.sh` gives the repo back to you.

## Two overlapping deletion tools

`delete_episode.py` takes episode indices; `clean_dataset.py` finds the
episodes itself. Both reindex, but they are **not** equivalent:

- `delete_episode.py` rewrites `data/` and renames `videos/`, and does not know
  about auxiliary label trees. On a dataset carrying `heatmap_labels/` it will
  leave those files unrenamed, so every episode after the first deleted one ends
  up beside another episode's labels. It also does not refresh `total_chunks`.
- `clean_dataset.py` discovers auxiliary per-episode parquet trees (any
  sibling directory with a `chunk-000/`) and moves them in lockstep.

Prefer `clean_dataset.py` when the dataset has label trees, or fix the
labels afterwards.
