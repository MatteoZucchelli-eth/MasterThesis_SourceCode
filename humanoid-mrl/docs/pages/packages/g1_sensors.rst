G1 Sensors
==========

This package contains development camera publishers, the mocap bridge, and the
head-camera wall-plane estimator.

Wall-plane estimator
--------------------

The estimator fits the drilling wall from aligned D435 depth and publishes the
plane, fit points, and an RViz marker:

.. code-block:: bash

   ros2 launch g1_sensors wall_plane.launch.py

Simulation publishes head depth by default. On hardware,
docker/head_camera.yaml enables aligned depth at 640x480 and 15 Hz:

.. code-block:: bash

   ros2 topic hz /camera/d435/aligned_depth_to_color/image_raw

The planner uses the fitted plane when vla_planner_depth_source is stereo.
Configure the estimator through wall_plane.launch.py; depth_bias_m applies a
measured range correction. Geometry helpers live in g1_common.plane and
g1_common.depth.
