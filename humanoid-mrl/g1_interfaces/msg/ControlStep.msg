# Synchronized state and action snapshot for GR00T VLA training

# Proprioceptive state
unitree_hg/LowState low_state

# Current FK end-effector poses: xyz (3) + rot6d (6).
# Left is the palm; RIGHT IS THE DRILL TIP, not the palm — the tip is what must land
# on the target square, and its position error maps 1:1 onto the task error, whereas
# a palm pose folds a ~0.20 m lever arm into every orientation error.
float32[9] left_eef_pose
float32[9] right_eef_pose

# IK target end-effector poses (same frames as above): xyz (3) + rot6d (6)
float32[9] left_eef_target
float32[9] right_eef_target

# Base Twist Command: lin_x, lat_y, ang_z
float32[3] vel_cmd

# Task metric: pose of the red_target IN THE DRILL-TIP FRAME, i.e.
# T_tip^-1 * T_target, encoded as xyz (3) + rot6d (6) like every other 9D pose
# in this pipeline. Being tip-relative makes it egocentric and invariant to the
# base yaw. Decomposes into exactly the quantities a policy needs:
#   xyz          - tip -> target displacement (tip frame); its direction is
#                  where to move, its norm is tip_target_dist
#   rot6d        - a genuine rotation (3 DOF): how the drill must rotate to sit
#                  square-on to the wall. Well-posed to regress as a 6D; a bare
#                  direction is only 2 DOF and has no canonical rotation.
# Recorded per frame as a training label (an auxiliary head predicts it), never
# as a policy input: the target pose is only known from TF in sim, so feeding it
# in would let the model bypass the camera it is meant to learn from.
# tip_target_valid is false while the red_target TF has not been received.
bool tip_target_valid
float32[9] tip_target_9d
float32 tip_target_dist
