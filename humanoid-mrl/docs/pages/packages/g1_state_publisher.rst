G1 State Publisher
==================

The state publisher converts Unitree LowState messages into
sensor_msgs/JointState. ROS robot_state_publisher then publishes TF from the
robot URDF.

.. code-block:: bash

   ros2 launch g1_state_publisher state_publisher.launch.py
