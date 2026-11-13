Data Collection
===============

Parallel workers
----------------

Each worker runs one headless Isaac Sim and one control/recording stack in an
isolated Compose project.

.. code-block:: bash

   scripts/collect_workers.sh start 4
   scripts/collect_workers.sh status
   scripts/collect_workers.sh logs 1 stack
   scripts/collect_workers.sh stop

Workers write g1_dataset_worker_<n> directories in the repository. Use a fresh
output directory for each collection campaign.

Common environment options:

========================  =========================================
Option                    Purpose
========================  =========================================
DOMAIN_BASE               First ROS domain; default 40
SIM_CAMERA_PUBLISH_HZ     Camera rate; default 30
SIM_GUI=1                 Show one worker in a window
SIM_MEM_LIMIT             Isaac Sim memory limit
STACK_MEM_LIMIT           Control-stack memory limit
DRILL_CONTACT=1           Enable drill-bit collision
SIM_EXTRA_ARGS            Additional Isaac Sim arguments
========================  =========================================

Choose the worker count so the combined memory limits fit in host RAM. The
watchdog stops all workers if available host memory remains below its threshold.

Timing constraint
-----------------

The camera period must contain a whole number of 300 Hz physics steps. The
default 30 Hz rate uses ten steps. Invalid rates are rejected by
g1_simulation.config.render_period_s.

Collection nodes use simulation time, so machine load affects wall-clock
duration without changing the commanded motion duration. Do not enable
use_sim_time on hardware, where no /clock publisher exists.

Preflight checks
----------------

Before a long collection:

1. Confirm the wrist camera has one publisher.

   .. code-block:: bash

      ros2 topic info /camera/hand/color/image_raw

2. Watch recorder logs for stale images, state clamping, repeated frames, or
   failed automatic quality checks.

3. Check several saved episodes.

   .. code-block:: bash

      python -m dataset_tools.heatmap_labels.sync_check \
        --dataset <dataset> --episodes 0 1 2

4. Build labels only after synchronization passes.

   .. code-block:: bash

      python -m dataset_tools.heatmap_labels.process_dataset \
        --dataset <dataset>

Merge workers
-------------

Merge worker datasets after collection:

.. code-block:: bash

   dataset_tools/scripts/merge_datasets.sh --out g1_dataset_merged g1_dataset_worker_*

The command reindexes episodes, frames, tasks, and videos. Sources are unchanged,
and the output directory must not already exist.

Clean a dataset
---------------

Scan a merged dataset for slip episodes and all-black camera streams inside the
control container:

.. code-block:: bash

   dataset_tools/scripts/clean_dataset.sh <dataset> --dry-run

The scan checks every episode and writes ``rejected_episodes.csv`` in the
dataset. Review it, then delete the detected episodes and reindex the remaining
data:

.. code-block:: bash

   dataset_tools/scripts/clean_dataset.sh <dataset> --yes

Deletion is permanent. The cleaner keeps videos, parquet files, metadata, and
auxiliary per-episode label trees aligned while reindexing.

Per-episode analysis
--------------------

Run the visual checks inside the development container started by make control-sim:

.. code-block:: bash

   dataset_tools/scripts/data_analysis.sh data/small_targets/g1_dataset_merged

COUNT, SEED, STRIDE, and SYNC_ALL control sampling. Results are written below
dataset_checks/.

Dashboard
---------

Generate the dashboard with:

.. code-block:: bash

   dataset_tools/scripts/generate_dashboard.sh
   dataset_tools/scripts/generate_dashboard.sh /path/to/data

Run it in the control container. With no argument, it scans ``$DATA_ROOT`` or
the repository's ``data/`` directory. Outputs are written under
``data_analysis/``. Set ``OPEN_DASHBOARD=0`` to skip opening the dashboard.
