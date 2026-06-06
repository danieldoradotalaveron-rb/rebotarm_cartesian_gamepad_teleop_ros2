"""RViz markers for Cartesian teleop validation (visualization only)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from geometry_msgs.msg import Point, Pose, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class TeleopVizConfig:
    trail_max_samples: int = 200
    trail_min_step_m: float = 0.0002
    axis_length_m: float = 0.04
    axis_line_width_m: float = 0.0025
    sphere_diameter_m: float = 0.008


@dataclass
class TcpTrailState:
    max_samples: int
    min_step_m: float
    points: deque[tuple[float, float, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.points = deque(maxlen=self.max_samples)

    def reset(self) -> None:
        self.points.clear()

    def maybe_append(self, x: float, y: float, z: float) -> None:
        if self.points:
            lx, ly, lz = self.points[-1]
            dx, dy, dz = x - lx, y - ly, z - lz
            if math.sqrt(dx * dx + dy * dy + dz * dz) < self.min_step_m:
                return
        self.points.append((x, y, z))


def _color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def _pose_position(pose: Pose) -> np.ndarray:
    return np.array(
        [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
        dtype=np.float64,
    )


def _rotation_matrix_from_pose(pose: Pose) -> np.ndarray:
    from tf_transformations import quaternion_matrix

    quat = [
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]
    mat = quaternion_matrix(quat)
    return np.asarray(mat[:3, :3], dtype=np.float64)


def _sphere_marker(
    *,
    marker_id: int,
    ns: str,
    pose: Pose,
    diameter: float,
    color: ColorRGBA,
    frame_id: str,
) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose = pose
    marker.scale = Vector3(x=diameter, y=diameter, z=diameter)
    marker.color = color
    return marker


def _pose_axis_line_marker(
    *,
    marker_id: int,
    ns: str,
    pose: Pose,
    axis_length: float,
    axis_line_width: float,
    frame_id: str,
    alpha: float,
) -> Marker:
    origin = _pose_position(pose)
    rot = _rotation_matrix_from_pose(pose)
    axis_colors = (
        (1.0, 0.2, 0.2),
        (0.2, 0.9, 0.2),
        (0.2, 0.4, 1.0),
    )

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = float(axis_line_width)

    for axis_idx, rgb in enumerate(axis_colors):
        end = origin + rot[:, axis_idx] * axis_length
        for point, color_rgb in (
            (origin, rgb),
            (end, rgb),
        ):
            p = Point()
            p.x, p.y, p.z = (float(v) for v in point)
            marker.points.append(p)
            marker.colors.append(_color(*color_rgb, alpha))

    return marker


def _trail_marker(
    *,
    marker_id: int,
    trail_points: deque[tuple[float, float, float]],
    frame_id: str,
) -> Marker | None:
    if len(trail_points) < 2:
        return None

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = "teleop_trail"
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.003
    marker.color = _color(0.2, 0.6, 1.0, 0.85)

    for x, y, z in trail_points:
        p = Point()
        p.x, p.y, p.z = float(x), float(y), float(z)
        marker.points.append(p)

    return marker


def build_teleop_marker_array(
    *,
    current_pose: Pose,
    target_pose: Pose,
    trail_points: deque[tuple[float, float, float]],
    config: TeleopVizConfig,
    frame_id: str = "base_link",
    stamp=None,
) -> MarkerArray:
    """Build markers for current TCP, target TCP, local EE axes, and TCP trail."""
    array = MarkerArray()

    current_sphere = _sphere_marker(
        marker_id=0,
        ns="teleop_current",
        pose=current_pose,
        diameter=config.sphere_diameter_m,
        color=_color(0.1, 0.95, 0.2, 0.95),
        frame_id=frame_id,
    )
    target_sphere = _sphere_marker(
        marker_id=1,
        ns="teleop_target",
        pose=target_pose,
        diameter=config.sphere_diameter_m,
        color=_color(1.0, 0.55, 0.05, 0.95),
        frame_id=frame_id,
    )
    current_axes = _pose_axis_line_marker(
        marker_id=2,
        ns="teleop_current_axes",
        pose=current_pose,
        axis_length=config.axis_length_m,
        axis_line_width=config.axis_line_width_m,
        frame_id=frame_id,
        alpha=0.95,
    )
    target_axes = _pose_axis_line_marker(
        marker_id=3,
        ns="teleop_target_axes",
        pose=target_pose,
        axis_length=config.axis_length_m * 0.85,
        axis_line_width=config.axis_line_width_m,
        frame_id=frame_id,
        alpha=0.7,
    )

    if stamp is not None:
        for marker in (current_sphere, target_sphere, current_axes, target_axes):
            marker.header.stamp = stamp

    array.markers.extend([current_sphere, target_sphere, current_axes, target_axes])

    trail = _trail_marker(marker_id=4, trail_points=trail_points, frame_id=frame_id)
    if trail is not None:
        if stamp is not None:
            trail.header.stamp = stamp
        array.markers.append(trail)

    return array


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from rebotarm_msgs.msg import CartesianJogState

    rclpy.init()
    node = Node("teleop_viz_markers")

    node.declare_parameter("cartesian_jog_state_topic", "/rebotarm/cartesian_jog_state")
    node.declare_parameter("markers_topic", "/rebotarm/teleop_viz/markers")
    node.declare_parameter("fixed_frame", "base_link")
    node.declare_parameter("trail_max_samples", 200)
    node.declare_parameter("trail_min_step_m", 0.0002)
    node.declare_parameter("axis_length_m", 0.04)
    node.declare_parameter("axis_line_width_m", 0.0025)
    node.declare_parameter("sphere_diameter_m", 0.008)

    state_topic = str(node.get_parameter("cartesian_jog_state_topic").value)
    markers_topic = str(node.get_parameter("markers_topic").value)
    fixed_frame = str(node.get_parameter("fixed_frame").value)
    viz_config = TeleopVizConfig(
        trail_max_samples=int(node.get_parameter("trail_max_samples").value),
        trail_min_step_m=float(node.get_parameter("trail_min_step_m").value),
        axis_length_m=float(node.get_parameter("axis_length_m").value),
        axis_line_width_m=float(node.get_parameter("axis_line_width_m").value),
        sphere_diameter_m=float(node.get_parameter("sphere_diameter_m").value),
    )
    trail = TcpTrailState(viz_config.trail_max_samples, viz_config.trail_min_step_m)
    publisher = node.create_publisher(MarkerArray, markers_topic, 10)

    def on_state(msg: CartesianJogState) -> None:
        trail.maybe_append(
            float(msg.current_pose.position.x),
            float(msg.current_pose.position.y),
            float(msg.current_pose.position.z),
        )
        stamp = msg.header.stamp if msg.header.stamp.sec or msg.header.stamp.nanosec else None
        if stamp is None:
            stamp = node.get_clock().now().to_msg()
        marker_array = build_teleop_marker_array(
            current_pose=msg.current_pose,
            target_pose=msg.target_pose,
            trail_points=trail.points,
            config=viz_config,
            frame_id=fixed_frame,
            stamp=stamp,
        )
        publisher.publish(marker_array)

    node.create_subscription(CartesianJogState, state_topic, on_state, 10)
    node.get_logger().info(f"Teleop viz markers: state={state_topic} -> {markers_topic}")
    node.get_logger().info(f"Fixed frame: {fixed_frame} (base_link axes = grid XY plane)")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
