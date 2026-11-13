Hardware
========

Connections
-----------

Use the upper-right USB-C port for DisplayPort output.

The internal Ethernet addresses are:

================  ===================
Device            Address
================  ===================
Developer PC      192.168.123.164
Locked PC         192.168.123.161
Livox MID360      192.168.123.120
================  ===================

Initial access
--------------

Connect over Ethernet:

.. code-block:: bash

   ssh unitree@192.168.123.164

The factory password is 123. Remove the host ROS selection prompt from ~/.bashrc;
ROS runs inside the project containers.

Allow the user to run Docker, then log in again:

.. code-block:: bash

   sudo usermod -aG docker $USER

Wi-Fi
-----

Enable Wi-Fi and open the network UI:

.. code-block:: bash

   sudo nmcli radio wifi on
   nmtui

The wired interface must retain access to the G1 internal network. Configure the
wireless interface separately for remote access.

Jetson recovery and flashing
----------------------------

.. warning::

   Flashing erases storage. Obtain the current G1 Jetson firmware and system
   image from Unitree support and follow the instructions bundled with that
   release. Confirm the target device before writing an image.

To enter recovery mode:

1. Power on the G1.
2. Hold Power and REC for about two seconds.
3. Release Power, wait two seconds, then release REC.
4. Confirm that lsusb lists an NVIDIA APX device.

.. image:: ../_static/img/g1_jetson_recovery_buttons.png
   :alt: Power and REC buttons
   :width: 800px
   :align: center

NVIDIA publishes current JetPack releases and SDK Manager here:

* https://developer.nvidia.com/embedded/jetpack-archive
* https://developer.nvidia.com/sdk-manager

JetPack includes Docker integration. Verify Docker and the NVIDIA runtime after
flashing before building the deployment image.

Changing the drill bit
----------------------

The bit length is a kinematic parameter, not just geometry: ``drill_tip_link``
is the IK target and the recorded end-effector frame, so the whole stack steers
whatever that frame says. Nothing syncs the URDF and the Isaac asset for the bit
(unlike the cameras, which ``IsaacSimCfg.sync_camera_extrinsics`` keeps in
step), so a bit swap has to be applied by hand in all of these:

#. ``g1_state_publisher/urdf/g1_29_dof_with_hand_drill.urdf.xml`` --
   ``drill_tip_joint`` origin x, the bit length measured from the chuck face.
#. ``g1_state_publisher/urdf/meshes/right_drill_bit_hex.STL`` -- the shank span,
   currently x 0 to 0.040.
#. ``g1_state_publisher/urdf/meshes/right_drill_bit_tip.STL`` -- the tip span,
   currently x 0.040 to 0.0444 at radius 0.002.
#. ``g1_simulation/assets/isaacsim/g1/g1_29dof_rev_1_0_w_hilti_drill.usd`` --
   the two Mesh prims under ``hand_drill_hilti/drill_bit``.
#. ``g1_simulation/g1_simulation/config.py`` -- ``IsaacSimCfg.reset_position``
   x, by the same amount the bit length changed.

Measure from the chuck face. It is the one datum a caliper can reach on the
assembled drill, and every length above is stated from it.

``reset_position`` is on the list because the pelvis is pinned and the wall is
fixed: forward kinematics on the READY pose puts ``drill_tip_link`` 0.4545 m
ahead of the pelvis, so x = 2.80 leaves the tip 54.5 mm clear of the wall at
3.309 before the arm moves. A longer bit that does not move the base starts
inside the wall. A 0.0887 m longer bit was fitted once and needed x = 2.7113.

Two things that do *not* need editing. ``DRILL_BIT_COLLIDER_CUTOFF_X`` is keyed
on geometry rather than on prim names, so it survives a swap; and
``right_drill_hand.STL`` carries no bit, so the RViz drill is cosmetically short
by design.

The offset can be cross-checked optically, and it is worth doing once: find the
tip pixel in a wrist frame and back-project it down the bit axis through the
camera's calibrated pose. Expect to have to explain any disagreement rather than
average it away: one such check came out 20.7 mm short of the caliper (111.4
against a measured 132.11 mm in the drill body frame) with neither lens
distortion nor the axis assumption accounting for it -- the tip pixel sat 58 px
from the principal point, where distortion is sub-pixel, and the projection
reproduced the observed bit axis exactly -- which points at a photographed bit
that is not the measured one, or a caliper figure referenced to a different
origin. Treat the caliper as authoritative and the projection as the check.

Reference
---------

Unitree G1 developer documentation:
https://support.unitree.com/home/en/G1_developer/quick_development
