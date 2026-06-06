"""RViz validation target spheres for Cartesian teleop (visualization only)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from geometry_msgs.msg import Pose
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

PALE_BLUE_COLOR = ColorRGBA(r=0.55, g=0.78, b=0.95, a=0.55)
HIT_RED_COLOR = ColorRGBA(r=0.92, g=0.22, b=0.18, a=0.85)
VALIDATION_TARGET_NS = "teleop_validation_target"


@dataclass(frozen=True)
class ValidationTarget:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ValidationTargetsConfig:
    enabled: bool
    radius_m: float
    persistent_hit: bool
    targets: tuple[ValidationTarget, ...]


def parse_validation_targets(raw_targets) -> tuple[ValidationTarget, ...]:
    """Parse targets from ROS params.

    ROS2 param YAML cannot use nested sequences (``- [x, y, z]``). Use a flat
    ``[x0, y0, z0, x1, y1, z1, ...]`` list in config files.
    """
    if not raw_targets:
        return ()

    if isinstance(raw_targets[0], (list, tuple)):
        targets: list[ValidationTarget] = []
        for item in raw_targets:
            if len(item) != 3:
                raise ValueError(f"Each validation target must be [x, y, z], got {item!r}")
            targets.append(
                ValidationTarget(float(item[0]), float(item[1]), float(item[2]))
            )
        return tuple(targets)

    flat = [float(v) for v in raw_targets]
    if len(flat) % 3 != 0:
        raise ValueError(
            f"validation_targets flat list length must be a multiple of 3, got {len(flat)}"
        )
    return tuple(
        ValidationTarget(flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat), 3)
    )


def tcp_distance_to_target(
    tcp_x: float,
    tcp_y: float,
    tcp_z: float,
    target: ValidationTarget,
) -> float:
    dx = tcp_x - target.x
    dy = tcp_y - target.y
    dz = tcp_z - target.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def is_target_hit(distance_m: float, radius_m: float) -> bool:
    return distance_m <= radius_m


@dataclass
class ValidationHitTracker:
    persistent_hit: bool
    hit_flags: list[bool]

    @classmethod
    def create(cls, target_count: int, *, persistent_hit: bool) -> ValidationHitTracker:
        return cls(persistent_hit=persistent_hit, hit_flags=[False] * target_count)

    def update(
        self,
        tcp_x: float,
        tcp_y: float,
        tcp_z: float,
        targets: tuple[ValidationTarget, ...],
        radius_m: float,
    ) -> list[bool]:
        for index, target in enumerate(targets):
            in_radius = is_target_hit(
                tcp_distance_to_target(tcp_x, tcp_y, tcp_z, target),
                radius_m,
            )
            if in_radius:
                self.hit_flags[index] = True
            elif not self.persistent_hit:
                self.hit_flags[index] = False
        return list(self.hit_flags)


def _target_pose(target: ValidationTarget) -> Pose:
    pose = Pose()
    pose.position.x = target.x
    pose.position.y = target.y
    pose.position.z = target.z
    pose.orientation.w = 1.0
    return pose


def build_validation_target_markers(
    *,
    targets: tuple[ValidationTarget, ...],
    hit_flags: list[bool],
    radius_m: float,
    frame_id: str = "base_link",
    stamp=None,
) -> MarkerArray:
    array = MarkerArray()
    diameter = float(radius_m) * 2.0

    for index, target in enumerate(targets):
        marker = Marker()
        marker.header.frame_id = frame_id
        if stamp is not None:
            marker.header.stamp = stamp
        marker.ns = VALIDATION_TARGET_NS
        marker.id = index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = _target_pose(target)
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = diameter
        marker.color = HIT_RED_COLOR if hit_flags[index] else PALE_BLUE_COLOR
        array.markers.append(marker)

    return array


def build_validation_target_marker_array(
    *,
    config: ValidationTargetsConfig,
    tcp_x: float,
    tcp_y: float,
    tcp_z: float,
    hit_tracker: ValidationHitTracker,
    frame_id: str = "base_link",
    stamp=None,
) -> MarkerArray:
    if not config.enabled or not config.targets:
        return MarkerArray()

    hit_flags = hit_tracker.update(tcp_x, tcp_y, tcp_z, config.targets, config.radius_m)
    return build_validation_target_markers(
        targets=config.targets,
        hit_flags=hit_flags,
        radius_m=config.radius_m,
        frame_id=frame_id,
        stamp=stamp,
    )


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from rebotarm_msgs.msg import CartesianJogState

    rclpy.init()
    node = Node("teleop_validation_targets")

    default_targets = [
        0.24, -0.16, 0.22, 0.24, 0.16, 0.22, 0.34, -0.16, 0.22, 0.34, 0.16, 0.22,
        0.44, 0.00, 0.22, 0.28, 0.00, 0.10, 0.38, -0.10, 0.10, 0.38, 0.10, 0.10,
        0.30, -0.10, 0.34, 0.30, 0.10, 0.34, 0.40, 0.00, 0.34, 0.44, 0.00, 0.14,
    ]
    node.declare_parameter("cartesian_jog_state_topic", "/rebotarm/cartesian_jog_state")
    node.declare_parameter("markers_topic", "/rebotarm/teleop_viz/validation_targets")
    node.declare_parameter("fixed_frame", "base_link")
    node.declare_parameter("validation_targets_enabled", True)
    node.declare_parameter("validation_target_radius_m", 0.015)
    node.declare_parameter("validation_target_persistent_hit", False)
    node.declare_parameter("validation_targets", default_targets)

    state_topic = str(node.get_parameter("cartesian_jog_state_topic").value)
    markers_topic = str(node.get_parameter("markers_topic").value)
    fixed_frame = str(node.get_parameter("fixed_frame").value)
    config = ValidationTargetsConfig(
        enabled=bool(node.get_parameter("validation_targets_enabled").value),
        radius_m=float(node.get_parameter("validation_target_radius_m").value),
        persistent_hit=bool(node.get_parameter("validation_target_persistent_hit").value),
        targets=parse_validation_targets(node.get_parameter("validation_targets").value),
    )
    hit_tracker = ValidationHitTracker.create(
        len(config.targets),
        persistent_hit=config.persistent_hit,
    )
    publisher = node.create_publisher(MarkerArray, markers_topic, 10)

    def on_state(msg: CartesianJogState) -> None:
        stamp = msg.header.stamp if msg.header.stamp.sec or msg.header.stamp.nanosec else None
        if stamp is None:
            stamp = node.get_clock().now().to_msg()
        marker_array = build_validation_target_marker_array(
            config=config,
            tcp_x=float(msg.current_pose.position.x),
            tcp_y=float(msg.current_pose.position.y),
            tcp_z=float(msg.current_pose.position.z),
            hit_tracker=hit_tracker,
            frame_id=fixed_frame,
            stamp=stamp,
        )
        publisher.publish(marker_array)

    node.create_subscription(CartesianJogState, state_topic, on_state, 10)
    node.get_logger().info(
        f"Validation targets: enabled={config.enabled} count={len(config.targets)} "
        f"radius={config.radius_m:.3f}m persistent_hit={config.persistent_hit}"
    )
    node.get_logger().info(f"Publishing markers on {markers_topic} (frame={fixed_frame})")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
