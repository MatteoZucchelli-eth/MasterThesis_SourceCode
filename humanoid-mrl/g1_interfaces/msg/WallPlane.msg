# A plane fitted to a depth sensor's view of the wall being drilled, and how
# well it fitted. Published by g1_sensors' wall_plane_node; consumed by
# g1_control's VLA_PLANNER as the range half of its target estimate.
#
# The fit quality travels WITH the geometry rather than on a separate topic
# because the consumer's question is never "where is the plane" but "where is
# the plane, and may I trust it". Split across two messages the two can be
# paired up wrongly, and a stale inlier count vouching for a fresh plane reads
# as a good measurement.

# header.frame_id is the frame `point` and `normal` are expressed in, and it is
# NOT the camera's: the node resolves the plane into a body frame (pelvis by
# default) at the depth frame's own stamp. That frame is static while the wall
# is, so a consumer that has to reach back to an earlier snapshot -- and the
# depth donor does, since detections and depth frames never share a tick -- can
# do it with the arm anchor it already holds instead of a second TF lookup.
std_msgs/Header header

# A point on the plane: the centroid of the points fitted, which is where the
# fit is best determined -- the plane's angular uncertainty pivots about it, so
# a small error in `normal` moves the surface least nearby.
geometry_msgs/Point point

# Unit normal, pointing OUT of the wall towards the camera that measured it.
# The drill travels along its negation.
geometry_msgs/Vector3 normal

# Points within the fit tolerance of this plane.
int32 inliers

# Points the fit was offered, after decimation and depth banding. inliers/sampled
# is the fraction of the view that is this plane, which is what separates a wall
# filling the frame from a lucky triple through a cluttered scene.
int32 sampled

# RMS distance of the inliers from the plane [m] -- the sensor's noise on this
# surface. On a wall known to be flat this is also the diagnostic for a depth
# stream carrying ray length instead of z: it goes from millimetres to
# centimetres (see g1_simulation camera_graph._add_depth_publisher).
float32 rms_m
