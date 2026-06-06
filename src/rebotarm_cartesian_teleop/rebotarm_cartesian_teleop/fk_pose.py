"""FK pose conversion (geometry_msgs) — no hardware side effects."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Pose
from tf_transformations import quaternion_from_matrix, quaternion_matrix


def fk_arrays_to_pose(position: np.ndarray, rotation: np.ndarray) -> Pose:
    mat = np.eye(4)
    mat[:3, :3] = rotation
    quat = quaternion_from_matrix(mat)

    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose


def pose_to_rotation_matrix(pose: Pose) -> np.ndarray:
    quat = [
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]
    mat = quaternion_matrix(quat)
    return np.asarray(mat[:3, :3], dtype=np.float64)
