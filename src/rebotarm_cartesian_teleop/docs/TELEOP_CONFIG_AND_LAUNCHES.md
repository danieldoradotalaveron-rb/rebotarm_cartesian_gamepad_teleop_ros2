# Teleop configuration and launches

Stable Cartesian gamepad teleop stack: local-window jog, base (joint1) jog, dry-run IK,
fake joint states for RViz, and rigid D405 eye-in-hand TF under `end_link`.

## Stable architecture

```
joy_node  →  joy_cartesian_mapper  →  /rebotarm/cartesian_jog_cmd
                                              ↓
                                    cartesian_jog_core
                                              ↓
                         /rebotarm/cartesian_jog_state
                         /rebotarm/fake_joint_states
                                              ↓
                              robot_state_publisher  →  /tf, /robot_description
                                              ↓
                         teleop_viz_markers, teleop_validation_targets, RViz
```

Teleop nodes load `config/cartesian_teleop.yaml`. TF/URDF launches load
`rebotarm_bringup/robot_description_launch.py` (xacro expansion + D405 launch args).

### Pure core vs ROS shell

| Module | Role |
|--------|------|
| `joy_mapping.py` | Pure Joy → `CartesianJogCmd` |
| `jog_core_logic.py` | State machine, local window, base jog, IK orchestration, **post-IK gate sequence** |
| `cartesian_params.py` | ROS parameter declare/load → typed config bundle |
| `cartesian_jog_core.py` | ROS shell: topics, timers, logging, `tick()` orchestration |
| `fk_kinematics.py` / `ik_kinematics.py` | FK/IK adapters over SDK |
| `fake_joint_state.py` | Fake `JointState` builder |

Post-IK safety gates run in fixed order via `apply_ik_gate_sequence()` in
`jog_core_logic.py` (called from `cartesian_jog_core.tick()` after `solve_target_ik`):

1. `JOINT1_GLOBAL_OPERATIONAL_LIMIT`
2. `JOINT1_ANCHOR_WINDOW`
3. `JOINT_NEAR_LIMIT`
4. `IK_NO_EFFECT`

First failing gate wins. Solver-level rejections (`JOINT_DELTA_TOO_LARGE`, `IK_ERROR_TOO_HIGH`)
remain inside `solve_target_ik` before this sequence.

**Tests:** gate ordering in driver fork
`integration/rebotarm_cartesian_teleop/test/test_ik_gate_sequence.py`.
Individual joint1 gate helpers: `test_joint1_gates.py`. Parameter loading (unit):
`test/unit/test_cartesian_params.py`.

## Config section ownership

Single file: `rebotarm_cartesian_teleop/config/cartesian_teleop.yaml`.

| Section | Node | Contents |
|---------|------|----------|
| `input_mapping` | `joy_cartesian_mapper` | Axis indices, deadzone, deadman, speed scale, smoothing, base-jog axis mapping |
| `teleop_geometry` | `cartesian_jog_core` | Workspace, local-window bounds, `initial_q`, `ee_frame`, initial target pose |
| `ik_and_safety` | `cartesian_jog_core` | IK mode/tolerance/iterations, joint deltas, joint1 anchor/global gates, IK_NO_EFFECT thresholds |
| `output_and_sim` | `cartesian_jog_core` | `dry_run`, `output_mode`, fake joint state topic/rate, `servo_hz`, command timeout |
| `diagnostics` | `cartesian_jog_core` | IK quality log interval, limit/drift warning margins |
| `visualization` | `teleop_viz_markers`, `teleop_validation_targets` | Marker topics, `fixed_frame`, validation sphere layout |

D405 mount/TF (not in `cartesian_teleop.yaml`):

| File | Role |
|------|------|
| `rebotarm_bringup/rebotarm_bringup/robot_description_launch.py` | Runtime D405 launch args (`enable_d405`, `d405_mount_xyz`, `d405_mount_rpy`, …) |
| `rebotarm_bringup/config/d405_mount.yaml` | Reference mount values (not loaded by launches) |
| `rebotarm_bringup/docs/D405_EYE_IN_HAND.md` | D405 link/joint chain and TF validation commands |

### Future config profiles (not implemented)

Base YAML + launch-level override files.

| Profile | Purpose |
|---------|---------|
| `cartesian_teleop.yaml` | Stable default sim/RViz (current) |
| `cartesian_teleop_hardware_safe.yaml` | Future hardware override |
| `cartesian_teleop_isaac.yaml` | Future Isaac Sim override |
| `d405_mount.yaml` | D405 mount/TF reference |

## Message contracts

### `CartesianJogState` (`rebotarm_msgs/msg/CartesianJogState.msg`)

- `q_current`, `q_target`: six elements, order `joint1` … `joint6` (radians).
- Matches `sensor_msgs/JointState` names published on `/rebotarm/fake_joint_states`.

### `CartesianJogCmd` (`rebotarm_msgs/msg/CartesianJogCmd.msg`)

- `command_frame_kind`: frame semantics for linear jog input.
  - `local_window_frame` — local-window teleop (`enable_local_teleop_window: true`).
  - `base_link` — base-frame jog (`enable_local_teleop_window: false`).
- `base_jog_active`: when true, `cartesian_jog_core` integrates joint1 velocity and skips Cartesian IK for that tick.
- `joint1_jog_velocity_rad_s`: joint1 velocity command while `base_jog_active` is true.

## Launch matrix

| ID | Launch file | Nodes started | RSP | RViz | Joint states expected | Safe with |
|----|-------------|---------------|-----|------|----------------------|-----------|
| A | *(just recipes)* `run-joy`, `run-joy-mapper`, `run-cartesian-core` | `joy_node`, `joy_cartesian_mapper`, `cartesian_jog_core` | No | No | Publishes `/rebotarm/fake_joint_states` | B, C (one RSP total) |
| B | `cartesian_teleop_validation_rviz.launch.py` | RSP, `teleop_viz_markers`, `teleop_validation_targets`, `rviz2` | Yes | Yes | `/rebotarm/fake_joint_states` | A, C |
| C | `cartesian_teleop_gripper_rviz.launch.py` | `rviz2` only (`rviz2_gripper_view`) | No | Yes | TF from B or another RSP | A, B |
| D | `rebotarm_bringup/d405_tf_diagnostics.launch.py` | RSP, `joint_state_publisher` (zeros) | Yes | No | `/rebotarm/fake_joint_states` or static zeros | Standalone TF check only |
| — | `fake_robot_state_publisher.launch.py` | RSP | Yes | No | `/rebotarm/fake_joint_states` | A (not with B/D/sim_rviz) |
| — | `cartesian_teleop_sim_rviz.launch.py` | RSP, `rviz2` | Yes | Yes | `/rebotarm/fake_joint_states` | A (not with B) |
| E | `rebotarm_bringup/bringup.launch.py` | `reBotArmController`, RSP, optional RViz | Yes | Optional | `/rebotarm/joint_states` (real) | Hardware only |
| — | `rebotarm_bringup/rviz.launch.py` | RSP, `rviz2` | Yes | Yes | `/rebotarm/joint_states` | Hardware only |
| — | `rebotarm_bringup/driver_only.launch.py` | `reBotArmController` | No | No | Publishes real joint states | Hardware only |
| F | *(future)* hardware safe teleop | Safety bridge | — | — | Real | Not implemented |
| G | *(experimental)* Gazebo | — | — | — | — | Not in repo |

### Duplicate `robot_state_publisher` risk

One RSP instance per robot description. Do **not** run together:

- B + D, B + `fake_robot_state_publisher`, B + `cartesian_teleop_sim_rviz`
- B/E + any second RSP on the same URDF

Duplicate RSP nodes publish competing `/robot_description` and `/tf`.

### Shared launch helper

`rebotarm_bringup/robot_description_launch.py`:

- `d405_launch_arguments()` — D405 xacro launch args
- `robot_description_parameter()` — expanded `reBot-DevArm_fixend.xacro`

RSP node construction is duplicated per launch file (no shared RSP helper).

## How to run

### Teleop validation (main stable visual test)

Terminal 1–3:

```bash
just run-joy
just run-joy-mapper
just run-cartesian-core
```

Terminal 4:

```bash
just run-teleop-validation-rviz
```

RViz saved view: `TeleopBaseValidation` (`cartesian_teleop_validation.rviz`, Fixed Frame `base_link`).

### Gripper / operator RViz (second window)

With validation stack (B) already running:

```bash
just run-teleop-gripper-rviz
```

RViz config: `cartesian_teleop_gripper_view.rviz`.

- Saved view name: `GripperFollowD405`
- View type: `ThirdPersonFollower`
- Target Frame: `end_link`
- Fixed Frame: `base_link`

D405 optical frames: `d405_color_optical_frame`, `d405_depth_optical_frame` (not used as RViz Target Frame).

### D405 TF validation

```bash
ros2 launch rebotarm_bringup d405_tf_diagnostics.launch.py
ros2 run tf2_ros tf2_echo end_link d405_color_optical_frame
```

Mount overrides: `d405_mount_xyz`, `d405_mount_rpy`, … (see `rebotarm_bringup/docs/D405_EYE_IN_HAND.md`).

## RViz configs

| File | Package | Fixed Frame | Primary view |
|------|---------|-------------|--------------|
| `cartesian_teleop_validation.rviz` | `rebotarm_cartesian_teleop` | `base_link` | `TeleopBaseValidation` (Orbit) |
| `cartesian_teleop_gripper_view.rviz` | `rebotarm_cartesian_teleop` | `base_link` | `GripperFollowD405` (ThirdPersonFollower → `end_link`) |
| `rebotarm.rviz` | `rebotarm_bringup` | — | General arm view (hardware/sim_rviz) |

## Stale / reference-only items

| Item | Status |
|------|--------|
| Gazebo / `ros_gz` / `gazebo_eye_in_hand` | Not present in source |
| `d405_mount.yaml` | Reference only; launch defaults in `robot_description_launch.py` |
| `cartesian_joint1_window_rad` | Declared in `cartesian_jog_core` (default 0.25); not in YAML (`cartesian_joint1_window_warning_rad` is in YAML) |
| `base_jog_left_button` / `base_jog_right_button` | Mapper defaults; inactive while `base_jog_input_type: axis` |
| Build/install/log artifacts | Not tracked; `just clean` or `rm -rf build install log` |

## Out of scope

- Gazebo physics simulation
- Isaac Sim integration
- Hardware teleop safety bridge
- RealSense camera driver
- Hand-eye calibration
