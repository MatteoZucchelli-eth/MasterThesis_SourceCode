Deployment
==========

.. warning::

   Put the G1 in debug mode before sending commands. Press L2 + R2 and confirm
   that the eye LEDs turn green. L2 + A selects debug position mode.

The repository must already be present on the robot. Run all deployment commands
directly on the G1; the Makefile does not perform remote SSH operations.

Camera stack
------------

.. code-block:: bash

   make deployment
   make stop-deployment

The deployment profile starts:

* realsense-node for the head D435;
* g1-hand-cam-deploy for the See3CAM wrist camera;
* dev-container-deploy for the control stack.

Set HEAD_CAMERA_SERIAL in docker/.env.deploy when multiple RealSense cameras are
connected. Keep the leading underscore expected by the driver.

make deployment recreates both camera services so changes to
docker/head_camera.yaml take effect. Verify the streams before a run:

.. code-block:: bash

   ros2 topic hz /camera/d435/color/image_raw
   ros2 topic hz /camera/d435/aligned_depth_to_color/image_raw

Attach to dev-container-deploy and launch the required control stack, for example:

.. code-block:: bash

   ros2 launch g1_state_machine real_launch.py

Teleoperation
-------------

Use the profile matching the hardware:

.. code-block:: bash

   make teleoperation # workstation and Isaac Sim
   make teleop-real  # G1 cameras and headset server
   make stop-teleoperation
   make stop-teleop-real

teleoperation sends simulated images through the ZMQ bridges. teleop-real reads
the head and wrist ROS topics directly.

Hardware dependencies
---------------------

Deployment dependencies are listed in deploy_deps.repos and built into the
deployment image with vcstool.
