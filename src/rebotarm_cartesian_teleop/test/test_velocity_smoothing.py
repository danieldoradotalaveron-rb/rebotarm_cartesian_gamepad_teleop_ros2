"""Tests for linear velocity smoothing in joy_mapping."""

from __future__ import annotations

import pytest
from conftest import default_mapper_config, make_joy_with_defaults

from rebotarm_cartesian_teleop.joy_mapping import (
    VelocitySmoothingState,
    apply_linear_velocity_smoothing,
    map_joy_to_cmd,
    resolve_smoothing_dt_s,
    smooth_linear_axis,
)


def _smooth_cfg(**overrides):
    return default_mapper_config(
        enable_velocity_smoothing=True,
        max_linear_accel_m_s2=0.25,
        velocity_smoothing_reset_on_deadman_release=True,
        velocity_smoothing_reset_on_soft_stop=True,
        **overrides,
    )


def _map_smoothed(
    joy,
    cfg,
    state: VelocitySmoothingState,
    *,
    now_ns: int,
    last_publish_ns: int | None = None,
):
    state.last_publish_time_ns = last_publish_ns
    return map_joy_to_cmd(
        joy,
        cfg,
        latest_joy_time_ns=now_ns,
        now_ns=now_ns,
        smoothing_state=state,
        publish_hz=30.0,
    )


def test_smoothing_disabled_matches_previous_behavior():
    cfg = default_mapper_config(enable_velocity_smoothing=False)
    joy = make_joy_with_defaults(axis1=1.0, axis0=0.5, axis5=-0.5, deadman=True)
    state = VelocitySmoothingState(prev_linear_x=0.5)
    now_ns = int(0.1 * 1e9)
    cmd = _map_smoothed(joy, cfg, state, now_ns=now_ns, last_publish_ns=0)
    scale = cfg.max_linear_velocity * cfg.speed_scale_default
    assert cmd is not None
    assert cmd.linear.x == pytest.approx(1.0 * scale)
    assert cmd.linear.y == pytest.approx(0.5 * scale)
    assert cmd.linear.z == pytest.approx(-0.5 * scale)
    assert state.prev_linear_x == pytest.approx(0.5)


def test_first_tick_does_not_jump_to_max_velocity():
    cfg = _smooth_cfg(max_linear_velocity=0.06)
    joy = make_joy_with_defaults(axis1=1.0, deadman=True)
    state = VelocitySmoothingState()
    dt_ns = int((1.0 / 30.0) * 1e9)
    cmd = _map_smoothed(joy, cfg, state, now_ns=dt_ns, last_publish_ns=0)
    target = 0.06
    max_step = cfg.max_linear_accel_m_s2 / 30.0
    assert cmd is not None
    assert cmd.linear.x == pytest.approx(max_step)
    assert cmd.linear.x < target
    assert cmd.linear.x <= max_step + 1e-9


def test_velocity_increases_at_most_accel_times_dt_per_axis():
    cfg = _smooth_cfg()
    joy = make_joy_with_defaults(axis1=1.0, axis0=1.0, deadman=True)
    state = VelocitySmoothingState()
    dt_ns = int(0.05 * 1e9)
    cmd1 = _map_smoothed(joy, cfg, state, now_ns=dt_ns, last_publish_ns=0)
    cmd2 = _map_smoothed(joy, cfg, state, now_ns=2 * dt_ns, last_publish_ns=dt_ns)
    max_delta = cfg.max_linear_accel_m_s2 * 0.05
    assert cmd1 is not None and cmd2 is not None
    assert abs(cmd2.linear.x - cmd1.linear.x) <= max_delta + 1e-9
    assert abs(cmd2.linear.y - cmd1.linear.y) <= max_delta + 1e-9


def test_direction_reversal_ramps_through_zero():
    cfg = _smooth_cfg(max_linear_velocity=0.06)
    joy_pos = make_joy_with_defaults(axis1=1.0, deadman=True)
    joy_neg = make_joy_with_defaults(axis1=-1.0, deadman=True)
    state = VelocitySmoothingState()
    dt_ns = int((1.0 / 30.0) * 1e9)

    for tick in range(20):
        now = (tick + 1) * dt_ns
        prev = tick * dt_ns if tick > 0 else None
        _map_smoothed(joy_pos, cfg, state, now_ns=now, last_publish_ns=prev)

    assert state.prev_linear_x > 0.0
    max_step = cfg.max_linear_accel_m_s2 / 30.0
    prev_before_flip = state.prev_linear_x

    cmd_flip = _map_smoothed(
        joy_neg,
        cfg,
        state,
        now_ns=21 * dt_ns,
        last_publish_ns=20 * dt_ns,
    )
    assert cmd_flip is not None
    assert cmd_flip.linear.x == pytest.approx(prev_before_flip - max_step, rel=1e-6)
    assert cmd_flip.linear.x > 0.0

    velocities: list[float] = []
    for tick in range(30):
        now = (22 + tick) * dt_ns
        prev = (21 + tick) * dt_ns
        cmd = _map_smoothed(joy_neg, cfg, state, now_ns=now, last_publish_ns=prev)
        assert cmd is not None
        velocities.append(cmd.linear.x)

    assert any(v <= 0.0 for v in velocities)
    assert velocities[-1] == pytest.approx(-0.06, rel=0.05)


def test_deadman_release_zeros_immediately_and_resets_state():
    cfg = _smooth_cfg()
    joy_active = make_joy_with_defaults(axis1=1.0, deadman=True)
    joy_idle = make_joy_with_defaults(axis1=1.0, deadman=False)
    state = VelocitySmoothingState()
    dt_ns = int((1.0 / 30.0) * 1e9)

    for tick in range(10):
        now = (tick + 1) * dt_ns
        prev = tick * dt_ns if tick > 0 else None
        _map_smoothed(joy_active, cfg, state, now_ns=now, last_publish_ns=prev)
    assert state.prev_linear_x > 0.0

    cmd = _map_smoothed(
        joy_idle,
        cfg,
        state,
        now_ns=11 * dt_ns,
        last_publish_ns=10 * dt_ns,
    )
    assert cmd is not None
    assert cmd.linear.x == 0.0
    assert cmd.linear.y == 0.0
    assert cmd.linear.z == 0.0
    assert state.prev_linear_x == 0.0
    assert state.prev_linear_y == 0.0
    assert state.prev_linear_z == 0.0


def test_soft_stop_zeros_immediately_and_resets_state():
    cfg = _smooth_cfg()
    joy_active = make_joy_with_defaults(axis1=1.0, deadman=True)
    joy_stop = make_joy_with_defaults(axis1=1.0, deadman=True, soft_stop=True)
    state = VelocitySmoothingState()
    dt_ns = int((1.0 / 30.0) * 1e9)

    for tick in range(10):
        now = (tick + 1) * dt_ns
        prev = tick * dt_ns if tick > 0 else None
        _map_smoothed(joy_active, cfg, state, now_ns=now, last_publish_ns=prev)
    assert state.prev_linear_x > 0.0

    cmd = _map_smoothed(
        joy_stop,
        cfg,
        state,
        now_ns=11 * dt_ns,
        last_publish_ns=10 * dt_ns,
    )
    assert cmd is not None
    assert cmd.soft_stop is True
    assert cmd.linear.x == 0.0
    assert state.prev_linear_x == 0.0


def test_r1_boost_ramps_toward_higher_target():
    cfg = _smooth_cfg(max_linear_velocity=0.06)
    joy_normal = make_joy_with_defaults(axis1=1.0, deadman=True)
    joy_boost = make_joy_with_defaults(axis1=1.0, deadman=True, speed_boost=True)
    state = VelocitySmoothingState()
    dt_ns = int((1.0 / 30.0) * 1e9)

    for tick in range(30):
        now = (tick + 1) * dt_ns
        prev = tick * dt_ns if tick > 0 else None
        _map_smoothed(joy_normal, cfg, state, now_ns=now, last_publish_ns=prev)

    normal_v = state.prev_linear_x
    assert normal_v == pytest.approx(0.06, rel=0.05)

    cmd_boost = _map_smoothed(
        joy_boost,
        cfg,
        state,
        now_ns=31 * dt_ns,
        last_publish_ns=30 * dt_ns,
    )
    boosted_target = 0.06 * cfg.speed_scale_boost
    assert cmd_boost is not None
    assert cmd_boost.speed_scale == cfg.speed_scale_boost
    assert cmd_boost.linear.x > normal_v
    assert cmd_boost.linear.x < boosted_target


def test_resolve_smoothing_dt_fallback():
    assert resolve_smoothing_dt_s(0, None, 30.0) == pytest.approx(1.0 / 30.0)
    assert resolve_smoothing_dt_s(1_000_000_000, 0, 30.0, max_dt_s=0.5) == pytest.approx(1.0 / 30.0)
    assert resolve_smoothing_dt_s(50_000_000, 0, 30.0) == pytest.approx(0.05)


def test_smooth_linear_axis_clamps_delta():
    assert smooth_linear_axis(1.0, 0.0, 0.1) == pytest.approx(0.1)
    assert smooth_linear_axis(-1.0, 0.5, 0.1) == pytest.approx(0.4)


def test_apply_smoothing_deadman_off_immediate_zero():
    cfg = _smooth_cfg()
    state = VelocitySmoothingState(prev_linear_x=0.5)
    x, y, z = apply_linear_velocity_smoothing(
        0.06,
        0.0,
        0.0,
        deadman=False,
        soft_stop=False,
        cfg=cfg,
        state=state,
        dt_s=0.033,
    )
    assert (x, y, z) == (0.0, 0.0, 0.0)
    assert state.prev_linear_x == 0.0
