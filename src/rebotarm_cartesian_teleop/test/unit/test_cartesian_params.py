"""Unit tests for cartesian_params declare/defaults (no rebotarm_msgs)."""

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


def test_parameter_defaults_table_matches_declarations(rclpy_context):
    node = _make_node("test_cartesian_params_count")
    try:
        declare_cartesian_core_parameters(node)
        for name in CARTESIAN_CORE_PARAMETER_DEFAULTS:
            assert node.has_parameter(name)
            assert node.get_parameter(name).value == CARTESIAN_CORE_PARAMETER_DEFAULTS[name]
    finally:
        node.destroy_node()


def test_yaml_style_overrides_respected_on_declare(rclpy_context):
    overrides = [
        Parameter("servo_hz", Parameter.Type.DOUBLE, 25.0),
        Parameter("enable_local_teleop_window", Parameter.Type.BOOL, False),
        Parameter("ik_task_mode", Parameter.Type.STRING, "full_6d"),
    ]
    node = _make_node("test_cartesian_params_overrides", overrides=overrides)
    try:
        declare_cartesian_core_parameters(node)
        assert node.get_parameter("servo_hz").value == pytest.approx(25.0)
        assert node.get_parameter("enable_local_teleop_window").value is False
        assert node.get_parameter("ik_task_mode").value == "full_6d"
    finally:
        node.destroy_node()
