"""Tests for teleop RViz marker builder (visualization only)."""

from __future__ import annotations

from collections import deque

from geometry_msgs.msg import Pose

from rebotarm_cartesian_teleop.teleop_viz_markers import (
    TcpTrailState,
    TeleopVizConfig,
    build_teleop_marker_array,
)


def _pose(x: float, y: float, z: float) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.w = 1.0
    return pose


def test_build_markers_includes_current_target_and_trail():
    trail = deque([ (0.30, 0.0, 0.20), (0.31, 0.0, 0.20) ])
    array = build_teleop_marker_array(
        current_pose=_pose(0.31, 0.0, 0.20),
        target_pose=_pose(0.32, 0.0, 0.20),
        trail_points=trail,
        config=TeleopVizConfig(),
    )
    namespaces = {m.ns for m in array.markers}
    assert namespaces == {
        "teleop_current",
        "teleop_target",
        "teleop_current_axes",
        "teleop_target_axes",
        "teleop_trail",
    }
    assert all(m.header.frame_id == "base_link" for m in array.markers)


def test_trail_state_skips_duplicate_samples():
    trail = TcpTrailState(max_samples=10, min_step_m=0.001)
    trail.maybe_append(0.0, 0.0, 0.0)
    trail.maybe_append(0.0001, 0.0, 0.0)
    trail.maybe_append(0.002, 0.0, 0.0)
    assert len(trail.points) == 2


def test_trail_marker_omitted_when_single_point():
    array = build_teleop_marker_array(
        current_pose=_pose(0.30, 0.0, 0.20),
        target_pose=_pose(0.30, 0.0, 0.20),
        trail_points=deque([(0.30, 0.0, 0.20)]),
        config=TeleopVizConfig(),
    )
    assert "teleop_trail" not in {m.ns for m in array.markers}


def test_current_and_target_sphere_colors_differ():
    array = build_teleop_marker_array(
        current_pose=_pose(0.30, 0.0, 0.20),
        target_pose=_pose(0.31, 0.0, 0.20),
        trail_points=deque(),
        config=TeleopVizConfig(),
    )
    current = next(m for m in array.markers if m.ns == "teleop_current")
    target = next(m for m in array.markers if m.ns == "teleop_target")
    assert current.color.g > target.color.g
    assert target.color.r > current.color.r
