G1 Control
==========

g1_control owns the high-rate control loop and the controller selected by the
state machine.

Main components
---------------

* ControllerManagerNode reads RobotState and publishes Unitree LowCmd messages.
* DampingController and ReadyController handle safe startup states.
* WholeBodyRLController runs whole-body and upper-body control.
* RobotModel provides Pinocchio kinematics and joint indexing.
* VLAClient and VLAActionManager receive and buffer GR00T action chunks.
* DebugVizNode publishes optional RViz markers outside the control loop.

Configuration is under src/g1_control/config/. Launch the control stack with:

.. code-block:: bash

   ros2 launch g1_control control_launch.py

Use enable_debug_viz:=false when visualization is not needed.
