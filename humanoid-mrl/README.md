# humanoid-mrl

Simulation, control and evaluation code for a Unitree G1 humanoid with a drill
mounted on its right wrist. It is the robot side of the master thesis
*precise drilling with a VLA*. The companion repository
[Isaac-GR00T](../Isaac-GR00T) trains and serves the learned models.

The system, in one sentence: a fine-tuned GR00T VLA brings the drill close to
the target named in the instruction ("drill in the {color} {shape}"). A perception
model then finds that target in the wrist image, and a classical planner (ray
and wall-plane localisation, visual servoing, straight insertion) places the
tip with millimetre accuracy.

## Contents

- [1. Reproduce the thesis results](#1-reproduce-the-thesis-results)
- [2. Full pipeline (needs an NVIDIA GPU)](#2-full-pipeline-needs-an-nvidia-gpu)
- [3. Where each thesis component lives](#3-where-each-thesis-component-lives)
- [4. Play in simulation](#4-play-in-simulation)
- [5. Other workflows](#5-other-workflows)

---

## 1. Reproduce the thesis results

The recorded evaluation runs are in `metrics/`, one folder per system, with
one CSV per task. This command recomputes the success tables of the results
chapter from them:

```bash
scripts/thesis_results.sh
```

```
System         Contact     10mm     5mm   2.5mm
Planner        on         29/36   28/36   25/36
TTH            on         32/36   27/36    8/36
TTH            off        34/36   32/36   30/36
Vanilla-480    off        32/36   30/36   25/36
Vanilla-256    off        23/36   23/36   16/36
```

---

## 2. Full pipeline (needs an NVIDIA GPU)

**Requirements:**

- Linux
- an NVIDIA RTX GPU that can run Isaac Sim 5.1
- Docker with the NVIDIA Container Toolkit
- a GPU with at least 16 GB for serving the models

Training needs larger GPUs; see the Isaac-GR00T README.

The thesis pipeline has four steps:

| Step | What | Where |
|---|---|---|
| 2a | Collect scripted demonstrations in Isaac Sim | here |
| 2b | Merge, clean, label and balance the dataset | here |
| 2c | Fine-tune the VLA and the perception model | [Isaac-GR00T](../Isaac-GR00T/README.md#thesis-tip-target-head-and-target-perception) |
| 2d | Evaluate on the 36 test tasks | here, with the models served from Isaac-GR00T |

### 2a. Collect demonstrations

Each worker is a headless Isaac Sim plus the control and recording stack. The
`AUTO_COLLECT` state plans a curved path to a standoff in front of the
commanded target, then pushes straight in. It saves the episode once the tip
has held the target. Each worker writes its own dataset
`g1_dataset_worker_<i>/` at the repository root and loops until you stop it.

```bash
scripts/collect_workers.sh start 4    # 4 parallel workers
scripts/collect_workers.sh status     # episodes collected so far
SIM_GUI=1 scripts/collect_workers.sh start 1   # watch the robot (one worker)
scripts/collect_workers.sh stop
```

The thesis dataset was collected with the defaults:

- drill contact off
- six training targets; red triangle, green circle and blue square are held out
- parameters in [auto_collect.yaml](src/g1_control/config/auto_collect.yaml)

### 2b. Clean, label and balance the dataset

Run these in a shell in the `sim-base` container, from the repository root.
Files written from the container are owned by root; `scripts/fix_ownership.sh`
gives them back to you.

```bash
docker run --rm -it -v "$PWD":/workspace -w /workspace sim-base bash
source /opt/ros/jazzy/setup.bash && colcon build --packages-up-to g1_control   # once

# 1. merge the per-worker datasets into one
dataset_tools/scripts/merge_datasets.sh --out data/my_dataset g1_dataset_worker_*

# 2. remove episodes where the bit slipped on the wall, or with black frames
dataset_tools/scripts/clean_dataset.sh data/my_dataset --dry-run   # look first
dataset_tools/scripts/clean_dataset.sh data/my_dataset

# 3. project the target into every wrist frame (labels for the perception model)
dataset_tools/scripts/generate_pixel_labels.sh data/my_dataset

# 4. cap each 1 cm distance band at the median band size (perception training only)
dataset_tools/scripts/balance_labels.sh data/my_dataset --report-only   # look first
dataset_tools/scripts/balance_labels.sh data/my_dataset --apply
```

Optional checks:

```bash
dataset_tools/scripts/data_analysis.sh data/my_dataset   # trajectory, overlay and sync checks
dataset_tools/scripts/generate_dashboard.sh data         # dataset statistics dashboard
```

To push the dataset to the cluster for training, run
`dataset_tools/scripts/sync_to_euler.sh data/my_dataset` on the host. All
tools are described in [dataset_tools/README.md](dataset_tools/README.md).

### 2c. Train the models

See the [Isaac-GR00T README](../Isaac-GR00T/README.md#thesis-tip-target-head-and-target-perception).

### 2d. Evaluate

The evaluation runs the 12 scenarios S1–S12, three targets each, so 36 tasks.
It records tip and target poses to one CSV per task.

**1. Serve the models.** Run this from the Isaac-GR00T repository. The action
VLA goes on port 5555, and the perception model on port 5556 (needed for the
Planner only). This needs a `.env` there with a Hugging Face token; see
[Credentials](../Isaac-GR00T/README.md#1-credentials-the-env-file).

```bash
make up-d MODEL_PATH=<vla-checkpoint>        PORT=5555
make up-d MODEL_PATH=<perception-checkpoint> PORT=5556
```

**2. Run the evaluation.** Run this from this repository, with one command per
system of the thesis:

| System | Model on 5555 | Command |
|---|---|---|
| Planner | TTH (+ perception on 5556) | `ROLLOUT_STATE=vla_planner ENABLE_WALL_PLANE=1 ROLLOUT_S=120 scripts/evaluate_vla.sh start metrics/planner_contact` |
| TTH, contact | TTH | `ROLLOUT_S=120 scripts/evaluate_vla.sh start metrics/tth_contact` |
| TTH, no contact | TTH | `DRILL_CONTACT=0 ROLLOUT_S=120 scripts/evaluate_vla.sh start metrics/tth_nocontact` |
| Vanilla-480 | Vanilla, 480 px | `DRILL_CONTACT=0 ROLLOUT_S=120 scripts/evaluate_vla.sh start metrics/vanilla480` |
| Vanilla-256 | Vanilla, 256 px | `DRILL_CONTACT=0 ROLLOUT_S=120 scripts/evaluate_vla.sh start metrics/vanilla256` |

```bash
scripts/evaluate_vla.sh status metrics/tth_contact   # progress (x/36)
scripts/evaluate_vla.sh logs stack                   # controller logs
scripts/evaluate_vla.sh stop
scripts/thesis_results.sh metrics/tth_contact        # success counts when done
```

Useful options:

- `SIM_GUI=1` shows Isaac Sim and RViz.
- `EVAL_SCENARIO_FILTER="5,9:green-circle"` runs a subset.
- `scripts/evaluate_vla.sh --help` lists all options.

The VLA samples its actions, so a rerun can differ by a task or two.

---

## 3. Where each thesis component lives

| Thesis | Code |
|---|---|
| Fixed-base Isaac Sim scene, randomised targets, held-out cells | [g1_simulation/isaacsim/scene.py](src/g1_simulation/g1_simulation/isaacsim/scene.py) |
| Scripted collection (`AUTO_COLLECT`) and its parameters | [states/auto_collect.py](src/g1_control/g1_control/states/auto_collect.py), [auto_collect.yaml](src/g1_control/config/auto_collect.yaml) |
| Recorder (camera-synchronised rows, geometric labels) | [g1_teleoperation](src/g1_teleoperation) |
| Slip filter and dataset checks | [dataset_tools/clean_dataset.py](dataset_tools/clean_dataset.py), [data_analysis/dq.py](data_analysis/dq.py) |
| Wrist-image labels and distance balancing | [dataset_tools/heatmap_labels/](dataset_tools/heatmap_labels) |
| VLA client (chunking, smoothing, deadband) | [vla/client.py](src/g1_control/g1_control/vla/client.py), [vla_config.yaml](src/g1_control/config/vla_config.yaml) |
| Deployment pipeline (approach, handover, localisation, servo, insertion) | [states/vla_planner.py](src/g1_control/g1_control/states/vla_planner.py), [vla_planner.yaml](src/g1_control/config/vla_planner.yaml) |
| Handover gate / visual servo | [vla/handover.py](src/g1_control/g1_control/vla/handover.py), [vla/visual_servo.py](src/g1_control/g1_control/vla/visual_servo.py) |
| Wall-plane fit from head depth | [g1_sensors](src/g1_sensors) |
| Test scenarios S1–S12 | [isaacsim/eval_scenarios.py](src/g1_simulation/g1_simulation/isaacsim/eval_scenarios.py) |
| Success metric | [vla_success.py](src/g1_control/g1_control/vla_success.py) |

---

## 4. Play in simulation

**1. Serve the models** (from Isaac-GR00T; only needed for the VLA states):

```bash
make up-d MODEL_PATH=<vla-checkpoint>        PORT=5555
make up-d MODEL_PATH=<perception-checkpoint> PORT=5556
```

**2. Start Isaac Sim and the development container:**

```bash
xhost +local:docker            # keyboard input and the Isaac window
make control-sim               # fixed base, for drilling
make control-sim DRILL_CONTACT=1   # the drill bit touches the wall
make control-sim WALK=1        # free base, for walking (no drilling)
```

**3. Enter the container and launch the state machine:**

```bash
docker compose -f docker/docker-compose.yaml exec dev-container bash
colcon build && source install/setup.bash
ros2 launch g1_state_machine full_launch.py
```

**4. Change state** with the keyboard, which emulates the Unitree remote, or
from a second container shell:

| State | Keys | Command |
|---|---|---|
| Ready (default pose) | `Q` + `↑` | `ros2 topic pub --once /fsm_trigger std_msgs/msg/String "{data: ready}"` |
| VLA | `O` + `3` | `... "{data: vla}"` |
| VLA + planner | `Q` + `3` | `... "{data: vla_planner}"` |
| Scripted collection | `Q` + `1` | `... "{data: auto_collect}"` |
| Walk | `O` + `1` | `... "{data: move}"` |
| Damping (stop) | `Q` + `2` | `... "{data: damping}"` |

Set the target before entering a VLA state:

```bash
ros2 topic pub --once /task std_msgs/msg/String "{data: 'drill in the {color} {shape}'}"
```

**Walking** (`WALK=1`): enter Ready, then Walk, press `9` in the Isaac window
to release the elastic band, then drive with `W`/`A`/`S`/`D` and turn with
`J`/`L`.

**Isaac window keys:**

- `R`: reset the robot and randomise the targets
- `V`: next evaluation scenario
- `9`: toggle the elastic band

---

## 5. Other workflows

```bash
make simulation     # Isaac Sim with the robot
make control-sim    # development container + Isaac Sim
make control        # development/control container only
make teleoperation  # development, simulation, bridges and Teleimager
make deployment     # on the robot: the G1 deployment stack
make help           # all targets
```

The full documentation (hardware, deployment, packages) is built with
`make build-docs`.
