IsaacLab Training
=================

Training lives in the separate IsaacLab-MRL repository:

https://github.com/ethz-mrl/IsaacLab-MRL

1. Clone IsaacLab-MRL and the MRL rsl_rl fork.
2. Set RSL_RL_HOST_PATH in IsaacLab-MRL/docker/.env.base.
3. Start and enter the container:

   .. code-block:: bash

      cd IsaacLab-MRL/docker
      ./container.py start
      ./container.py enter

4. Train:

   .. code-block:: bash

      ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-G1-Blind-v0 --headless

5. Evaluate:

   .. code-block:: bash

      ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-G1-Blind-v0 --num_envs 1 \
        --checkpoint /workspace/isaaclab/logs/rsl_rl/g1_blind/data/checkpoint.pt

Copy the selected policy into src/g1_control/models/.
