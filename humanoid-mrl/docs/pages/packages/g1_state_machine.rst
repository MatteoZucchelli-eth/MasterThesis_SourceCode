G1 State Machine
================

G1FSM maps remote-control buttons and messages on /fsm_trigger to FSMState
values published on /fsm_state. It reads the Unitree remote from /lowstate.

Transitions
-----------

===================  =============  ====================
Trigger              Buttons        Destination
===================  =============  ====================
damping              L2 + B         DAMPING
ready                L2 + D-pad up  READY
move                 R2 + A         MOVE
vla                  R2 + X         VLA
teleoperation        R2 + Y         TELEOP
goal_pose            R2 + B         GOAL_POSE
auto_collect         L2 + A         AUTO_COLLECT
vla_planner          L2 + X         VLA_PLANNER
fk_calibration       R1 + X         FK_CALIBRATION
docs                 START          no state change
===================  =============  ====================

Transitions can also be requested directly:

.. code-block:: bash

   ros2 topic pub --once /fsm_trigger std_msgs/msg/String "{data: ready}"

Damping is available from every active state. Other working states require a
controller to be running first.

Goal pose
---------

GOAL_POSE applies a Cartesian offset in the current drill-tip frame:

.. code-block:: bash

   ros2 run g1_control goal_pose_publisher_node --x 0.05 --y 0.02 --z 0

FK calibration
--------------

Start the hardware calibration stack, enter FK_CALIBRATION with R1 + X, then
request each measurement round:

.. code-block:: bash

   ros2 launch g1_state_machine real_fk_calibration_launch.py
   ros2 topic pub --once /fk_calibration/start std_msgs/msg/Empty "{}"

Auto collection
---------------

AUTO_COLLECT follows three externally commanded phases:

.. code-block:: bash

   ros2 topic pub --once /auto_collect/phase std_msgs/String "{data: going_to_default}"
   ros2 topic pub --once /auto_collect/phase std_msgs/String "{data: trajectory_generation}"
   ros2 topic pub --once /auto_collect/phase std_msgs/String "{data: approaching}"

The current phase is published on /auto_collect/phase_status. Automatic recording
starts at APPROACHING, saves after the hold, and discards interrupted episodes.
Simulation reset clears the current plan and returns the arm to a fresh start
pose.
