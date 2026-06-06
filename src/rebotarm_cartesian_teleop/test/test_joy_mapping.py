"""Unit tests for joy_mapping pure logic."""

from __future__ import annotations

import pytest
from conftest import default_mapper_config, make_joy_with_defaults
from sensor_msgs.msg import Joy

from rebotarm_cartesian_teleop.joy_mapping import (
    axis_value,
    button_pressed,
    is_joy_fresh,
    map_joy_to_cmd,
)


def test_no_joy_received_does_not_publish():
    cfg = default_mapper_config()
    assert map_joy_to_cmd(None, cfg, latest_joy_time_ns=None, now_ns=0) is None


def test_deadman_not_pressed_zeros_linear():
    cfg = default_mapper_config()
    joy = make_joy_with_defaults(axis1=1.0, axis0=0.5, axis5=-0.5, deadman=False)
    cmd = map_joy_to_cmd(joy, cfg, latest_joy_time_ns=0, now_ns=0)
    assert cmd is not None
    assert cmd.deadman is False
    assert cmd.linear.x == 0.0
    assert cmd.linear.y == 0.0
    assert cmd.linear.z == 0.0


def test_deadman_pressed_scales_axes():
    cfg = default_mapper_config()
    joy = make_joy_with_defaults(axis1=1.0, axis0=0.5, axis5=-0.5, deadman=True)
    cmd = map_joy_to_cmd(joy, cfg, latest_joy_time_ns=0, now_ns=0)
    assert cmd is not None
    assert cmd.deadman is True
    scale = cfg.max_linear_velocity * cfg.speed_scale_default
    assert cmd.linear.x == pytest.approx(1.0 * scale)
    assert cmd.linear.y == pytest.approx(0.5 * scale)
    assert cmd.linear.z == pytest.approx(-0.5 * scale)


def test_deadzone_zeros_small_axis():
    cfg = default_mapper_config(deadzone=0.15)
    joy = make_joy_with_defaults(axis1=0.1, deadman=True)
    cmd = map_joy_to_cmd(joy, cfg, latest_joy_time_ns=0, now_ns=0)
    assert cmd is not None
    assert cmd.linear.x == 0.0


def test_soft_stop_zeros_linear():
    cfg = default_mapper_config()
    joy = make_joy_with_defaults(axis1=1.0, deadman=True, soft_stop=True)
    cmd = map_joy_to_cmd(joy, cfg, latest_joy_time_ns=0, now_ns=0)
    assert cmd is not None
    assert cmd.soft_stop is True
    assert cmd.linear.x == 0.0
    assert cmd.linear.y == 0.0
    assert cmd.linear.z == 0.0


def test_speed_boost_scales_velocity():
    cfg = default_mapper_config()
    joy = make_joy_with_defaults(axis1=1.0, deadman=True, speed_boost=True)
    cmd = map_joy_to_cmd(joy, cfg, latest_joy_time_ns=0, now_ns=0)
    assert cmd is not None
    assert cmd.speed_scale == cfg.speed_scale_boost
    scale = cfg.max_linear_velocity * cfg.speed_scale_boost
    assert cmd.linear.x == pytest.approx(scale)


def test_axis_inversion():
    cfg = default_mapper_config(invert_x=True, invert_y=True, invert_z=True)
    joy = make_joy_with_defaults(axis1=1.0, axis0=0.5, axis5=-0.5, deadman=True)
    cmd = map_joy_to_cmd(joy, cfg, latest_joy_time_ns=0, now_ns=0)
    assert cmd is not None
    scale = cfg.max_linear_velocity * cfg.speed_scale_default
    assert cmd.linear.x == pytest.approx(-1.0 * scale)
    assert cmd.linear.y == pytest.approx(-0.5 * scale)
    assert cmd.linear.z == pytest.approx(0.5 * scale)


def test_invalid_axis_and_button_indices_are_safe():
    cfg = default_mapper_config()
    joy = Joy()
    joy.axes = []
    joy.buttons = []
    assert button_pressed(joy, 4) is False
    assert axis_value(joy, 1, False, cfg.deadzone) == 0.0
    cmd = map_joy_to_cmd(joy, cfg, latest_joy_time_ns=0, now_ns=0)
    assert cmd is not None
    assert cmd.linear.x == 0.0


def test_joy_timeout_stops_publishing():
    cfg = default_mapper_config(joy_timeout_s=0.3)
    joy = make_joy_with_defaults(axis1=1.0, deadman=True)
    stale_now = int(0.5 * 1e9)
    assert map_joy_to_cmd(joy, cfg, latest_joy_time_ns=0, now_ns=0) is not None
    assert map_joy_to_cmd(joy, cfg, latest_joy_time_ns=0, now_ns=stale_now) is None


def test_is_joy_fresh_boundary():
    assert is_joy_fresh(None, 0, 0.3) is False
    assert is_joy_fresh(0, int(0.3 * 1e9), 0.3) is True
    assert is_joy_fresh(0, int(0.31 * 1e9), 0.3) is False
