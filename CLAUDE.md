# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`px4_llm_control` is a ROS 2 (ament_python) package that lets a user control a PX4
multicopter with plain-English instructions. An LLM (Claude, via the Anthropic API)
converts each instruction into a list of structured mission steps, which a state-machine
node executes over PX4 offboard control via uXRCE-DDS.

This package lives in a colcon workspace at `~/ros2_ws`; this directory is `~/ros2_ws/src/px4_llm_control`.

## Build & run

```bash
# Build (from the workspace root)
cd ~/ros2_ws
colcon build --packages-select px4_llm_control
source install/setup.bash
```

Running the full system requires three things, in order:

```bash
# 1. PX4 SITL
cd ~/PX4-Autopilot
make px4_sitl gz_x500

# 2. uXRCE-DDS bridge
MicroXRCEAgent udp4 -p 8888

# 3. LLM mission control (executor + GUI)
export ANTHROPIC_API_KEY=your_api_key_here
ros2 launch px4_llm_control px4_llm_control.launch.py
```

`command_gui` (Tkinter) and `command_cli` (stdin) are alternative front ends — both just
publish `/nl_command` strings and print `/nl_mission/status`. The launch file starts the
GUI; run `ros2 run px4_llm_control command_cli` instead/additionally for a terminal interface.

`NL_MISSION_MODEL` overrides the Claude model used by the planner (default `claude-sonnet-4-6`).

## Testing & linting

`package.xml` declares the standard ament test deps (`ament_copyright`, `ament_flake8`,
`ament_pep257`, `python3-pytest`), run via:

```bash
colcon test --packages-select px4_llm_control
```

Note: `test/` is currently empty — there are no `test_flake8.py` / `test_pep257.py` /
`test_copyright.py` files yet, so `colcon test` is currently a no-op for this package.

## Architecture

### Data flow

```
command_gui / command_cli  --/nl_command (String)-->  mission_executor
                            <--/nl_mission/status (String)--
```

`mission_executor.py` is the core node (`nl_mission_executor`). On each `/nl_command`
message it:
1. Snapshots the drone's current `x, y, z, heading` (NED).
2. Pushes `(instruction, snapshot)` onto a queue for a background **planner thread**
   (`_planner_worker`). The Anthropic API call blocks on the network and must never stall
   the 10 Hz offboard heartbeat, so it never runs on the main/tick thread.
3. The worker calls `LLMPlanner.plan()` (`llm_planner.py`), which forces Claude to call a
   `submit_mission_plan` tool and returns a validated list of step dicts
   (`action` ∈ `takeoff | goto | hold | velocity | attitude | land | rtl`, with NED
   `x/y/z`, `altitude`, `yaw`, `duration`, `vx/vy/vz`/`yawspeed`, or `roll/pitch/thrust`
   fields as appropriate).
4. Results are drained back on the main thread (`_drain_plans`) and appended to a
   `deque` of pending steps.

### State machine (`mission_executor.py`)

`_tick()` runs at `TICK_HZ = 10`. It always publishes `OffboardControlMode` (with the
`position`/`velocity`/`attitude` flag set based on the current `State`) plus a matching
setpoint, then dispatches based on `State`:

- `GROUNDED` — initial state (also re-entered after landing). Sits disarmed, streaming
  heartbeat position setpoints. Once `_steps` has at least one queued step (i.e. an
  instruction has been planned) **and** `HEARTBEAT_TICKS` of heartbeat have been sent,
  it arms and engages OFFBOARD mode (mirrors the arm/offboard sequence used by the
  sibling `interceptor_mission` package), then transitions to `IDLE` to dispatch that
  step. Arming/offboard is never automatic — it only happens in response to a queued
  instruction.
- `IDLE` — holds the last commanded position (`_hold_x/y/z`); pops and dispatches the
  next step from `_steps` if any are queued.
- `TAKEOFF` / `GOTO` — fly to `_goto_target` (x, y, z, yaw) via position setpoints;
  transition back to `IDLE` once within `POS_TOLERANCE_M`.
- `HOLD` — holds position until `_timer_until` (wall clock).
- `VELOCITY` — streams a `TrajectorySetpoint.velocity` (`_velocity_target`: vx, vy, vz,
  optional yawspeed) until `_timer_until`, then re-anchors `_hold_x/y/z` to wherever the
  vehicle ended up and returns to `IDLE`.
- `ATTITUDE` — streams a `VehicleAttitudeSetpoint` (`_attitude_target`: roll, pitch, yaw,
  thrust, converted to a quaternion + body thrust by `euler_to_quaternion()`) until
  `_timer_until`, then re-anchors `_hold_x/y/z` and returns to `IDLE`. This is an
  open-loop maneuver — no position/altitude hold while active.
- `LAND` / `RTL` — one-shot handoff to PX4's `AUTO_LAND` / `AUTO_RTL`, then move to
  `EXTERNAL_WAIT`.
- `EXTERNAL_WAIT` — waits for PX4 to disarm after a LAND/RTL, clears any remaining
  queued steps, then returns to `GROUNDED` (disarmed, not in OFFBOARD) until the next
  instruction is queued.

`velocity`/`attitude`/`hold` step values from the planner are clamped to safe ranges in
`_dispatch_next_step` (`MAX_VELOCITY_MPS`, `MAX_YAWSPEED_RADPS`, `MAX_TILT_RAD`,
`MIN_THRUST`/`MAX_THRUST`, `MAX_TIMED_STEP_S`) regardless of what the LLM returns.

Status/progress strings are published on `/nl_mission/status` via `_status_msg()` for
either front end to display, including a line whenever PX4's `arming_state`/`nav_state`
changes (`_cb_status`) — useful for diagnosing arm/offboard transitions during testing.

### Coordinate frame

All coordinates are PX4 local **NED**, matching `/fmu/out/vehicle_local_position_v1`:
`x` = North (m), `y` = East (m), `z` = Down (m) — negative `z` is above ground
(e.g. `z=-5` means 5 m up). Heading/yaw is radians clockwise from North. The LLM system
prompt (`llm_planner.py`) is responsible for resolving relative phrases ("forward",
"turn left", "climb") into these absolute NED values using the snapshot of current state.

### PX4 / uXRCE-DDS topics

- In: `/nl_command` (`std_msgs/String`)
- Out: `/nl_mission/status` (`std_msgs/String`)
- PX4 out (`/fmu/in/...`, BEST_EFFORT/TRANSIENT_LOCAL QoS): `offboard_control_mode`,
  `trajectory_setpoint`, `vehicle_attitude_setpoint`, `vehicle_command`
- PX4 in (`/fmu/out/...`, same QoS): `vehicle_local_position_v1`, `vehicle_status_v4`
