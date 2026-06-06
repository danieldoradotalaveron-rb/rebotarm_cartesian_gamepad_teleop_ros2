"""Tests for RViz validation target spheres (visualization only)."""

from __future__ import annotations

import pytest

from rebotarm_cartesian_teleop.teleop_validation_targets import (
    ValidationHitTracker,
    ValidationTarget,
    ValidationTargetsConfig,
    build_validation_target_marker_array,
    build_validation_target_markers,
    is_target_hit,
    parse_validation_targets,
    tcp_distance_to_target,
)

TARGET = ValidationTarget(0.30, 0.0, 0.27)
CONFIG = ValidationTargetsConfig(
    enabled=True,
    radius_m=0.025,
    persistent_hit=False,
    targets=(TARGET,),
)


def test_distance_below_radius_marks_target_hit():
    tracker = ValidationHitTracker.create(1, persistent_hit=False)
    flags = tracker.update(0.30, 0.0, 0.27, (TARGET,), 0.025)
    assert flags == [True]
    assert is_target_hit(tcp_distance_to_target(0.30, 0.01, 0.27, TARGET), 0.025)


def test_distance_above_radius_leaves_target_blue():
    tracker = ValidationHitTracker.create(1, persistent_hit=False)
    flags = tracker.update(0.30, 0.10, 0.27, (TARGET,), 0.025)
    assert flags == [False]


def test_persistent_hit_false_returns_to_blue_after_leaving():
    tracker = ValidationHitTracker.create(1, persistent_hit=False)
    tracker.update(0.30, 0.0, 0.27, (TARGET,), 0.025)
    flags = tracker.update(0.30, 0.10, 0.27, (TARGET,), 0.025)
    assert flags == [False]


def test_persistent_hit_true_keeps_target_red_after_first_contact():
    tracker = ValidationHitTracker.create(1, persistent_hit=True)
    tracker.update(0.30, 0.0, 0.27, (TARGET,), 0.025)
    flags = tracker.update(0.30, 0.10, 0.27, (TARGET,), 0.025)
    assert flags == [True]


def test_marker_count_matches_configured_targets():
    targets = parse_validation_targets(
        [
            [0.30, 0.00, 0.27],
            [0.35, 0.10, 0.27],
            [0.35, -0.10, 0.27],
        ]
    )
    array = build_validation_target_markers(
        targets=targets,
        hit_flags=[False, True, False],
        radius_m=0.025,
    )
    assert len(array.markers) == 3
    assert [m.id for m in array.markers] == [0, 1, 2]
    assert array.markers[1].color.r > 0.85
    assert array.markers[1].color.g < 0.35
    assert array.markers[0].color.b > 0.9
    assert array.markers[0].color.g > 0.7
    assert array.markers[2].color.b > 0.9


def test_disabled_mode_publishes_empty_marker_array():
    tracker = ValidationHitTracker.create(1, persistent_hit=False)
    config = ValidationTargetsConfig(
        enabled=False,
        radius_m=0.025,
        persistent_hit=False,
        targets=(TARGET,),
    )
    array = build_validation_target_marker_array(
        config=config,
        tcp_x=0.30,
        tcp_y=0.0,
        tcp_z=0.27,
        hit_tracker=tracker,
    )
    assert array.markers == []


def test_parse_validation_targets_flat_ros_params():
    targets = parse_validation_targets(
        [0.30, 0.00, 0.27, 0.35, 0.10, 0.27, 0.35, -0.10, 0.27]
    )
    assert len(targets) == 3
    assert targets[0] == ValidationTarget(0.30, 0.0, 0.27)
    assert targets[2] == ValidationTarget(0.35, -0.10, 0.27)


def test_parse_validation_targets_rejects_invalid_flat_length():
    with pytest.raises(ValueError, match="multiple of 3"):
        parse_validation_targets([0.30, 0.0])


def test_parse_validation_targets_rejects_invalid_nested_entry():
    with pytest.raises(ValueError, match="Each validation target"):
        parse_validation_targets([[0.30, 0.0]])
