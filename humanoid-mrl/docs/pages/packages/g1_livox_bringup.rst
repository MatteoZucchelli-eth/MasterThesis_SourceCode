G1 Livox Bringup
================

Launch the Livox MID360 driver:

.. code-block:: bash

   ros2 launch g1_livox_bringup g1_lidar_launch.py

Override its configuration when needed:

.. code-block:: bash

   ros2 launch g1_livox_bringup g1_lidar_launch.py user_config_path:=g1_mid360.json

Driver source: https://github.com/tu-darmstadt-ros-pkg/livox_ros_driver2
