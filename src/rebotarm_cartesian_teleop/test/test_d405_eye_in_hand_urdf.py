"""Static tests for the rigid D405 eye-in-hand robot description.

Validates xacro expansion of reBot-DevArm_fixend.xacro (rebotarm_bringup):
  - enable_d405:=true  adds the 4 D405 links + 4 fixed joints
  - enable_d405:=false adds zero d405_* frames
  - optical RPY uses the ROS optical convention
  - the 6 active arm joints are preserved
"""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

OPTICAL_RPY = "-1.5707963267948966 0 -1.5707963267948966"

D405_LINKS = [
    "d405_mount_link",
    "d405_camera_link",
    "d405_color_optical_frame",
    "d405_depth_optical_frame",
]
D405_JOINTS = [
    "d405_mount_joint",
    "d405_camera_body_joint",
    "d405_color_optical_joint",
    "d405_depth_optical_joint",
]
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def _xacro_path() -> str:
    from ament_index_python.packages import get_package_share_directory

    share = get_package_share_directory("rebotarm_bringup")
    return f"{share}/description/urdf/reBot-DevArm_fixend.xacro"


def _expand(enable_d405: bool | None = None) -> str:
    if shutil.which("xacro") is None:
        pytest.skip("xacro executable not found")
    cmd = ["xacro", _xacro_path()]
    if enable_d405 is not None:
        cmd.append(f"enable_d405:={'true' if enable_d405 else 'false'}")
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        pytest.skip("rebotarm_bringup not installed")


def _names(urdf: str, tag: str) -> set[str]:
    root = ET.fromstring(urdf)
    return {el.get("name") for el in root.findall(tag)}


def test_xacro_expands_with_d405():
    urdf = _expand(enable_d405=True)
    assert "<robot" in urdf


def test_enable_d405_true_adds_links_and_joints():
    urdf = _expand(enable_d405=True)
    links = _names(urdf, "link")
    joints = _names(urdf, "joint")
    for name in D405_LINKS:
        assert name in links, f"missing link {name}"
    for name in D405_JOINTS:
        assert name in joints, f"missing joint {name}"


def test_enable_d405_false_has_no_d405_frames():
    urdf = _expand(enable_d405=False)
    links = _names(urdf, "link")
    joints = _names(urdf, "joint")
    assert not any(name and "d405" in name for name in links)
    assert not any(name and "d405" in name for name in joints)


def test_default_includes_d405():
    urdf = _expand()
    assert "d405_color_optical_frame" in urdf


def test_optical_rpy_present():
    urdf = _expand(enable_d405=True)
    assert OPTICAL_RPY in urdf


def test_d405_joints_are_fixed():
    urdf = _expand(enable_d405=True)
    root = ET.fromstring(urdf)
    by_name = {j.get("name"): j for j in root.findall("joint")}
    for name in D405_JOINTS:
        assert by_name[name].get("type") == "fixed", f"{name} must be fixed"


def test_arm_active_joint_count_preserved():
    urdf = _expand(enable_d405=True)
    root = ET.fromstring(urdf)
    revolute = [j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"]
    assert sorted(revolute) == sorted(ARM_JOINTS)
    assert len(revolute) == 6


def test_mount_joint_parent_is_end_link():
    urdf = _expand(enable_d405=True)
    root = ET.fromstring(urdf)
    by_name = {j.get("name"): j for j in root.findall("joint")}
    mount = by_name["d405_mount_joint"]
    assert mount.find("parent").get("link") == "end_link"
