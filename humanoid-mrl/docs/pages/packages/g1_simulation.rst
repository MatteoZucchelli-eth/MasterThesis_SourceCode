G1 Simulation
=============

g1_simulation provides MuJoCo physics simulation and Isaac Sim simulation with
visual sensors.

.. code-block:: bash

   make simulation
   make simulation SIM_TYPE=sim-mujoco

IsaacSimCfg and MujocoCfg define backend options in
src/g1_simulation/g1_simulation/config.py. The camera period must contain a
whole number of physics steps; render_period_s validates it.
