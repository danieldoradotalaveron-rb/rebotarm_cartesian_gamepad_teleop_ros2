"""Unit tests for cartesian_params (ROS parameter loading)."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import pytest
import rclpy
from rclpy.parameter import Parameter

from rebotarm_cartesian_teleop.cartesian_params import (
    CARTESIAN_CORE_PARAMETER_DEFAULTS,
    declare_cartesian_core_parameters,
    load_cartesian_core_params,
)

if TYPE_CHECKING:
    from rclpy.node import Node


@pytest.fixture(scope="module")
def rclpy_context():
    log_dir = tempfile.mkdtemp(prefix="rebotarm_cartesian_params_test_")
    os.environ.setdefault("ROS_LOG_DIR", log_dir)
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def _make_node(name: str, *, overrides: list[Parameter] | None = None) -> Node:
    return rclpy.create_node(name, parameter_overrides=overrides or [])


def test_all_parameters_declared(rclpy_context):
    node = _make_node("test_cartesian_params_declared")
    try:
        declare_cartesian_core_parameters(node)
        for name in CARTESIAN_CORE_PARAMETER_DEFAULTS:
            assert node.has_parameter(name), f"missing parameter: {name}"
    finally:
        node.destroy_node()


def test_default_values_match_previous_behavior(rclpy_context):
    node = _make_node("test_cartesian_params_defaults")
    try:
        params = load_cartesian_core_params(node)

        assert params.core.cmd_topic == "/rebotarm/cartesian_jog_cmd"
        assert params.core.state_topic == "/rebotarm/cartesian_jog_state"
        assert params.core.output_mode == "dry_run"
        assert params.core.dry_run is True
        assert params.core.command_timeout_s == pytest.approx(0.3)
        assert params.core.servo_hz == pytest.approx(50.0)
        assert params.core.ee_frame == "end_link"

        assert params.geometry.initial_x == pytest.approx(0.30)
        assert params.geometry.initial_y == pytest.approx(0.00)
        assert params.geometry.initial_z == pytest.approx(0.20)
        assert params.geometry.urdf_path == ""
        assert params.geometry.initial_q == [0.0, -0.3, -0.3, 0.0, 0.0, 0.0]
        assert params.geometry.enable_local_teleop_window is True
        assert params.geometry.workspace.x_min == pytest.approx(0.150)
        assert params.geometry.workspace.x_max == pytest.approx(0.450)
        assert params.geometry.workspace.z_max == pytest.approx(0.450)

        assert params.base_jog.enable_base_joint_jog is True
        assert params.base_jog.base_joint_jog_speed_rad_s == pytest.approx(0.5)

        assert params.geometry.local_window_limits.x_min == pytest.approx(-0.12)
        assert params.geometry.local_window_limits.x_max == pytest.approx(0.18)
        assert params.geometry.local_window_limits.global_z_min == pytest.approx(0.020)
        assert params.geometry.local_window_limits.global_z_max == pytest.approx(0.450)

        assert params.ik.ik_config.max_iterations == 100
        assert params.ik.ik_config.tolerance == pytest.approx(0.001)
        assert params.ik.ik_config.task_mode == "position_only"
        assert params.ik.ik_config.max_ik_error == pytest.approx(0.005)
        assert params.ik.ik_config.max_joint_delta_rad == pytest.approx(0.25)
        assert params.ik.ik_failure_log_interval_s == pytest.approx(1.0)
        assert params.ik.candidate_drift_log_threshold_m == pytest.approx(0.001)

        assert params.safety.joint_limit_reject_config.reject_margin_rad == pytest.approx(0.05)
        assert params.safety.ik_no_effect_config.candidate_step_min_m == pytest.approx(0.0005)
        assert params.safety.joint1_global_cap_config.enabled is True
        assert params.safety.joint1_global_cap_config.min_rad == pytest.approx(-1.60)
        assert params.safety.joint1_global_cap_config.max_rad == pytest.approx(1.60)
        assert params.safety.joint1_anchor_window_config.enabled is True
        assert params.safety.joint1_anchor_window_config.hard_window_rad == pytest.approx(1.20)

        diag = params.diagnostics.ik_quality_log_config
        assert diag.joint_limit_warn_margin_rad == pytest.approx(0.35)
        assert diag.enable_cartesian_joint1_window_diagnostics is True
        assert diag.cartesian_joint1_window_warning_rad == pytest.approx(0.25)
        assert diag.joint1_large_delta_from_anchor_rad == pytest.approx(0.15)
        assert params.diagnostics.ik_quality_log_interval_s == pytest.approx(1.0)

        assert params.output.publish_fake_joint_states is True
        assert params.output.fake_joint_states_topic == "/rebotarm/fake_joint_states"
        assert params.output.fake_joint_state_hz == pytest.approx(50.0)
    finally:
        node.destroy_node()


def test_yaml_style_overrides_respected(rclpy_context):
    overrides = [
        Parameter("servo_hz", Parameter.Type.DOUBLE, 25.0),
        Parameter("enable_local_teleop_window", Parameter.Type.BOOL, False),
        Parameter("local_window_x_max_m", Parameter.Type.DOUBLE, 0.25),
        Parameter("base_joint_jog_speed_rad_s", Parameter.Type.DOUBLE, 0.75),
        Parameter("ik_task_mode", Parameter.Type.STRING, "full_6d"),
        Parameter("max_joint_delta_rad", Parameter.Type.DOUBLE, 0.10),
        Parameter("joint1_global_operational_min_rad", Parameter.Type.DOUBLE, -1.0),
        Parameter("enable_joint1_anchor_hard_gate", Parameter.Type.BOOL, False),
        Parameter("publish_fake_joint_states", Parameter.Type.BOOL, False),
        Parameter("fake_joint_state_hz", Parameter.Type.DOUBLE, 10.0),
    ]
    node = _make_node("test_cartesian_params_overrides", overrides=overrides)
    try:
        params = load_cartesian_core_params(node)

        assert params.core.servo_hz == pytest.approx(25.0)
        assert params.geometry.enable_local_teleop_window is False
        assert params.geometry.local_window_limits.x_max == pytest.approx(0.25)
        assert params.base_jog.base_joint_jog_speed_rad_s == pytest.approx(0.75)
        assert params.ik.ik_config.task_mode == "full_6d"
        assert params.ik.ik_config.max_joint_delta_rad == pytest.approx(0.10)
        assert params.safety.joint1_global_cap_config.min_rad == pytest.approx(-1.0)
        assert params.safety.joint1_anchor_window_config.enabled is False
        assert params.output.publish_fake_joint_states is False
        assert params.output.fake_joint_state_hz == pytest.approx(10.0)
    finally:
        node.destroy_node()


def test_joint1_warning_window_legacy_alias(rclpy_context):
    overrides = [
        Parameter("cartesian_joint1_window_warning_rad", Parameter.Type.DOUBLE, 0.25),
        Parameter("cartesian_joint1_window_rad", Parameter.Type.DOUBLE, 0.40),
    ]
    node = _make_node("test_cartesian_params_joint1_alias", overrides=overrides)
    try:
        params = load_cartesian_core_params(node)
        assert params.diagnostics.ik_quality_log_config.cartesian_joint1_window_warning_rad == (
            pytest.approx(0.40)
        )
    finally:
        node.destroy_node()


def test_parameter_defaults_table_matches_declarations(rclpy_context):
    node = _make_node("test_cartesian_params_count")
    try:
        declare_cartesian_core_parameters(node)
        for name in CARTESIAN_CORE_PARAMETER_DEFAULTS:
            assert node.has_parameter(name)
            assert node.get_parameter(name).value == CARTESIAN_CORE_PARAMETER_DEFAULTS[name]
    finally:
        node.destroy_node()
