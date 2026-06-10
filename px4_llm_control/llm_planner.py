#!/usr/bin/env python3
"""Converts natural-language mission instructions into structured PX4 mission steps.

Requires `pip install anthropic` and an ANTHROPIC_API_KEY in the environment.
Uses Claude's tool-use to force a schema-validated mission plan as output.
"""

import math
import os

from anthropic import Anthropic

DEFAULT_MODEL = os.environ.get('NL_MISSION_MODEL', 'claude-sonnet-4-6')

# All coordinates are PX4 local NED, matching /fmu/out/vehicle_local_position_v1:
#   x = North (m), y = East (m), z = Down (m) — negative z is above the ground
#   (e.g. z=-5 means 5 m up). Heading is radians clockwise from North.
SYSTEM_PROMPT = """You are a flight planner for a PX4 multicopter operating in NED coordinates:
  x = North (metres), y = East (metres), z = Down (metres) — negative z is above the ground
  (e.g. z=-5 means 5 m above ground). Heading is in radians, clockwise from North
  (0 = North, pi/2 = East, pi/-pi = South, -pi/2 = West).

You are given the drone's current state and a natural-language instruction. Call
submit_mission_plan with the ordered list of mission steps that implements it.

Resolve relative phrases ("go forward 5 metres", "turn left", "climb 3 metres", "come back")
using the drone's current position and heading, and convert them into absolute NED
coordinates / headings. The forward direction for heading h is the unit vector
(dx, dy) = (cos(h), sin(h)) in (x, y) — heading 0 (North) -> (+1, 0), pi/2 (East) ->
(0, +1), pi/-pi (South) -> (-1, 0), -pi/2 (West) -> (0, -1). "Forward"/"ahead" moves
along +(dx, dy); "backward"/"back" moves along -(dx, dy) — directly opposite of
forward, using whatever heading is current at that point in the plan (after any
preceding turns), never the original snapshot heading. "left" is the forward direction
for heading h-pi/2, and "right" is for heading h+pi/2. A "goto" that doesn't mention
altitude should keep the current z. Omit "yaw" when the instruction doesn't care about
heading.

The instruction may describe several actions in sequence ("go forward 1 m then back 1 m",
"turn right 90 degrees and go backward 3 metres", comma- or "then"-separated clauses,
etc.) — emit one step per action, in order. Track a running (x, y, z, heading) starting
from "Current state" and update it after each step you plan; use this UPDATED state, not
the original snapshot, to resolve relative phrases for later steps in the same plan. A
step that only changes heading ("turn left/right N degrees") with no translation should
be a "goto" at the same x/y/z with the new "yaw" — and that new heading is what later
"forward"/"backward"/"left"/"right" phrases in the plan are relative to.

Example: from x=0, y=0, z=-5, heading=0 (North), "turn right 90 degrees and go backward
3 metres" becomes two steps: (1) goto x=0, y=0, z=-5, yaw=pi/2 (now facing East); (2)
"backward" relative to facing East is West, i.e. -y, so goto x=0, y=-3, z=-5, yaw=pi/2.

Two additional step types give finer control:

- "velocity": command a velocity (vx, vy, vz in NED m/s) and optional yawspeed (rad/s,
  same clockwise-from-North sign convention as heading) for "duration" seconds. Use this
  whenever the instruction gives an explicit speed, e.g. "fly forward at 2 m/s for 5
  seconds" — resolve "forward"/"left"/"right" into vx/vy using the current heading, same
  as for goto.
- "attitude": command a body tilt (roll, pitch in radians, FRD frame), an absolute target
  "yaw" (radians, same convention as heading), and a normalized "thrust" (0..1, where
  ~0.5 roughly hovers the default SITL vehicle) for "duration" seconds. This is a raw,
  open-loop maneuver with no position or altitude hold — only use it when the instruction
  explicitly asks for a tilt/bank/roll/pitch/thrust maneuver, keep roll/pitch small
  (well under 0.35 rad) and "duration" short (a few seconds) unless told otherwise, since
  the vehicle will drift in position and altitude for the whole duration.
"""

PLAN_TOOL = {
    'name': 'submit_mission_plan',
    'description': 'Submit the ordered list of mission steps that implements the instruction.',
    'input_schema': {
        'type': 'object',
        'properties': {
            'steps': {
                'type': 'array',
                'minItems': 1,
                'items': {
                    'type': 'object',
                    'properties': {
                        'action': {
                            'type': 'string',
                            'enum': ['takeoff', 'goto', 'hold', 'velocity', 'attitude', 'land', 'rtl'],
                        },
                        'altitude': {
                            'type': 'number',
                            'description': 'takeoff: metres above ground, positive',
                        },
                        'x': {'type': 'number', 'description': 'goto: NED north, metres'},
                        'y': {'type': 'number', 'description': 'goto: NED east, metres'},
                        'z': {
                            'type': 'number',
                            'description': 'goto: NED down, metres (negative = above ground)',
                        },
                        'yaw': {
                            'type': 'number',
                            'description': (
                                'goto: optional target heading in radians. '
                                'attitude: required target heading in radians.'
                            ),
                        },
                        'vx': {'type': 'number', 'description': 'velocity: NED north speed, m/s'},
                        'vy': {'type': 'number', 'description': 'velocity: NED east speed, m/s'},
                        'vz': {
                            'type': 'number',
                            'description': 'velocity: NED down speed, m/s (negative = climbing)',
                        },
                        'yawspeed': {
                            'type': 'number',
                            'description': 'velocity: optional yaw rate, rad/s (clockwise from North)',
                        },
                        'roll': {'type': 'number', 'description': 'attitude: body roll, radians (FRD)'},
                        'pitch': {'type': 'number', 'description': 'attitude: body pitch, radians (FRD)'},
                        'thrust': {
                            'type': 'number',
                            'description': 'attitude: normalized thrust, 0..1 (~0.5 roughly hovers)',
                        },
                        'duration': {
                            'type': 'number',
                            'description': 'hold / velocity / attitude: seconds',
                        },
                    },
                    'required': ['action'],
                },
            },
        },
        'required': ['steps'],
    },
}

VALID_ACTIONS = {'takeoff', 'goto', 'hold', 'velocity', 'attitude', 'land', 'rtl'}
REQUIRED_FIELDS = {
    'takeoff': ('altitude',),
    'goto': ('x', 'y', 'z'),
    'hold': ('duration',),
    'velocity': ('vx', 'vy', 'vz', 'duration'),
    'attitude': ('roll', 'pitch', 'yaw', 'thrust', 'duration'),
    'land': (),
    'rtl': (),
}


class PlannerError(RuntimeError):
    """Raised when the LLM response can't be turned into a valid mission plan."""


class LLMPlanner:
    """Wraps the LLM call that turns one NL instruction into a list of mission steps."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self._client = Anthropic()
        self._model = model

    def plan(self, instruction: str, state: dict) -> list:
        """Return a validated list of mission-step dicts for `instruction`.

        `state` must provide the drone's current x/y/z (NED, metres) and heading
        (radians) so the model can resolve relative directions like "forward".
        """
        user_msg = (
            f"Current state: x={state['x']:.1f} m, y={state['y']:.1f} m, z={state['z']:.1f} m, "
            f"heading={state['heading']:.2f} rad ({math.degrees(state['heading']):.0f} deg)\n"
            f"Instruction: {instruction}"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[PLAN_TOOL],
            tool_choice={'type': 'tool', 'name': 'submit_mission_plan'},
            messages=[{'role': 'user', 'content': user_msg}],
        )
        return self._parse(response)

    @staticmethod
    def _parse(response) -> list:
        tool_use = next(
            (block for block in response.content if block.type == 'tool_use'), None)
        if tool_use is None:
            raise PlannerError(f'LLM did not call submit_mission_plan: {response.content}')

        steps = tool_use.input.get('steps')
        if not isinstance(steps, list) or not steps:
            raise PlannerError(f'LLM submitted no non-empty "steps" list: {tool_use.input}')

        for i, step in enumerate(steps):
            action = step.get('action')
            if action not in VALID_ACTIONS:
                raise PlannerError(f'Step {i} has unknown action {action!r}: {steps}')
            missing = [f for f in REQUIRED_FIELDS[action] if f not in step]
            if missing:
                raise PlannerError(f'Step {i} ({action}) is missing {missing}: {steps}')
            step.setdefault('yaw', None)

        return steps
