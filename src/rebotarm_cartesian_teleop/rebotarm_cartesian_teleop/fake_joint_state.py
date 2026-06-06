"""Simulation-only fake JointState helpers for RViz visualization."""

from __future__ import annotations

from sensor_msgs.msg import JointState

FAKE_JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)


def update_last_valid_fake_q(
    last_valid_fake_q: list[float] | None,
    q_current: list[float] | None,
    ik_success: bool,
    q_target: list[float],
) -> list[float]:
    """Track last valid IK solution for fake visualization; freeze on IK failure."""
    if ik_success and q_target:
        return [float(v) for v in q_target]
    if last_valid_fake_q is not None:
        return list(last_valid_fake_q)
    if q_current:
        return [float(v) for v in q_current]
    return []


def fake_joint_positions_to_publish(
    enabled: bool,
    last_valid_fake_q: list[float] | None,
    q_current: list[float] | None,
    ik_success: bool,
    q_target: list[float],
) -> tuple[list[float] | None, list[float] | None]:
    """Update fake-q state and return positions to publish (None when disabled)."""
    if not enabled:
        return last_valid_fake_q, None
    updated = update_last_valid_fake_q(last_valid_fake_q, q_current, ik_success, q_target)
    return updated, updated


def build_fake_joint_state(positions: list[float]) -> JointState:
    msg = JointState()
    msg.name = list(FAKE_JOINT_NAMES)
    msg.position = [float(v) for v in positions]
    return msg
