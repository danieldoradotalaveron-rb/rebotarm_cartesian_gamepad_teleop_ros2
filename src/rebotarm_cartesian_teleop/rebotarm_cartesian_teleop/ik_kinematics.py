"""Pure IK helper using reBotArm_control_py (no hardware, no retry)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .sdk_path import ensure_rebot_sdk_in_syspath


@dataclass
class IkSolveResult:
    success: bool
    q_target: list[float]
    error: float
    iterations: int
    reason: str


def _failure(reason: str, error: float = 0.0, iterations: int = 0) -> IkSolveResult:
    return IkSolveResult(
        success=False,
        q_target=[],
        error=error,
        iterations=iterations,
        reason=reason,
    )


def _clamp_config(model: Any, q: np.ndarray) -> np.ndarray:
    """Clamp q to joint limits (mirrors SDK inverse_kinematics._clamp_config)."""
    lo = np.array([
        float(x) if np.isfinite(x) else 0.0 for x in model.lowerPositionLimit
    ])
    hi = np.array([
        float(x) if np.isfinite(x) else 0.0 for x in model.upperPositionLimit
    ])
    clamped = np.maximum(q, lo)
    return np.minimum(clamped, hi)


def _position_error_local(
    T_cur: Any,
    target_position: np.ndarray,
) -> np.ndarray:
    """3D position error in LOCAL frame, consistent with LOCAL Jacobian rows."""
    import pinocchio as pin

    target_se3 = pin.SE3(T_cur.rotation, target_position)
    return pin.log6(T_cur.inverse() * target_se3).vector[:3]


def compute_ik_for_pose(
    model: Any,
    data: Any,
    end_frame_id: int,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    q_seed: np.ndarray,
    max_iterations: int,
    tolerance: float,
    max_ik_error: float,
) -> IkSolveResult:
    """Solve IK for a Cartesian target pose (deterministic, no retry).

    SDK ``result.success`` reflects convergence against ``ik_tolerance`` (strict).
    For dry-run Cartesian teleop we accept any solution whose final ``error`` is
    within ``max_ik_error``, even when the SDK marks ``success=False``. Hardware
    output remains disabled upstream.
    """
    ensure_rebot_sdk_in_syspath()
    from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, pos_rot_to_se3, solve_ik

    try:
        pos = np.asarray(target_position, dtype=np.float64).reshape(3)
        rot = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
        q_init = np.asarray(q_seed, dtype=np.float64).reshape(model.nq)
        target = pos_rot_to_se3(pos, rot)
        params = IKParams(max_iter=int(max_iterations), tolerance=float(tolerance))
        result = solve_ik(model, data, end_frame_id, target, q_init, params)
    except Exception:
        return _failure("IK_EXCEPTION")

    error = float(result.error)
    iterations = int(result.iterations)

    if len(result.q) != model.nq:
        return _failure("INVALID_IK_RESULT", error=error, iterations=iterations)

    if error > float(max_ik_error):
        return _failure("IK_ERROR_TOO_HIGH", error=error, iterations=iterations)

    return IkSolveResult(
        success=True,
        q_target=[float(v) for v in result.q],
        error=error,
        iterations=iterations,
        reason="",
    )


def compute_ik_for_position(
    model: Any,
    data: Any,
    end_frame_id: int,
    target_position: np.ndarray,
    q_seed: np.ndarray,
    max_iterations: int,
    tolerance: float,
    max_ik_error: float,
) -> IkSolveResult:
    """Solve position-only IK (3D task, orientation unconstrained).

    Mirrors the SDK damped least-squares CLIK loop but uses only the LOCAL
    position rows of the frame Jacobian. ``max_ik_error`` is interpreted as the
    position error norm (meters).
    """
    ensure_rebot_sdk_in_syspath()
    import pinocchio as pin
    from reBotArm_control_py.kinematics.inverse_kinematics import IKParams

    try:
        pos = np.asarray(target_position, dtype=np.float64).reshape(3)
        q = np.asarray(q_seed, dtype=np.float64).reshape(model.nq).copy()
        params = IKParams(max_iter=int(max_iterations), tolerance=float(tolerance))
    except Exception:
        return _failure("IK_EXCEPTION")

    prev_err = float("inf")
    iterations = 0

    for iteration in range(params.max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        T_cur = data.oMf[end_frame_id]
        err = _position_error_local(T_cur, pos)
        prev_err = float(np.linalg.norm(err))

        if prev_err < params.tolerance:
            return IkSolveResult(
                success=True,
                q_target=[float(v) for v in q],
                error=prev_err,
                iterations=iteration,
                reason="",
            )

        pin.computeJointJacobians(model, data, q)
        J = pin.getFrameJacobian(model, data, end_frame_id, pin.LOCAL)[:3, :]

        lam = params.damping * max(1.0, prev_err * 10.0)
        jjt = J @ J.T
        jjt[np.arange(3), np.arange(3)] += lam
        dq = params.step_size * J.T @ np.linalg.solve(jjt, err)

        alpha = 1.0
        for _ in range(4):
            q_new = _clamp_config(model, pin.integrate(model, q, alpha * dq))
            pin.forwardKinematics(model, data, q_new)
            pin.updateFramePlacements(model, data)
            T_new = data.oMf[end_frame_id]
            err_new = _position_error_local(T_new, pos)
            new_err = float(np.linalg.norm(err_new))
            if new_err < prev_err:
                q = q_new
                prev_err = new_err
                break
            alpha *= 0.5

        iterations = iteration + 1

    if len(q) != model.nq:
        return _failure("INVALID_IK_RESULT", error=prev_err, iterations=iterations)

    if prev_err > float(max_ik_error):
        return _failure("IK_ERROR_TOO_HIGH", error=prev_err, iterations=iterations)

    return IkSolveResult(
        success=True,
        q_target=[float(v) for v in q],
        error=prev_err,
        iterations=iterations,
        reason="",
    )


def orientation_drift_rad(rotation_from: np.ndarray, rotation_to: np.ndarray) -> float:
    """Angle (rad) between two rotation matrices (diagnostic only)."""
    ensure_rebot_sdk_in_syspath()
    import pinocchio as pin

    r_from = np.asarray(rotation_from, dtype=np.float64).reshape(3, 3)
    r_to = np.asarray(rotation_to, dtype=np.float64).reshape(3, 3)
    return float(np.linalg.norm(pin.log3(r_from.T @ r_to)))


def joint_delta_within_limit(
    q_target: list[float],
    q_seed: np.ndarray,
    max_joint_delta_rad: float,
) -> bool:
    q_t = np.asarray(q_target, dtype=np.float64)
    q_s = np.asarray(q_seed, dtype=np.float64).reshape(q_t.shape)
    return float(np.max(np.abs(q_t - q_s))) <= float(max_joint_delta_rad)
