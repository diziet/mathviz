"""L-system rewriting engine and 3D turtle interpreter.

Provides string rewriting for L-system grammars and a 3D turtle that
converts the resulting instruction string into line segments with
thickness information.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import Generator

logger = logging.getLogger(__name__)


@dataclass
class LSystemPreset:
    """Named L-system preset with axiom, rules, and defaults."""

    axiom: str
    rules: dict[str, str]
    default_angle: float
    default_iterations: int
    description: str


PRESETS: dict[str, LSystemPreset] = {
    "tree": LSystemPreset(
        axiom="F",
        rules={"F": "F[+F]F[-F]F"},
        default_angle=25.0,
        default_iterations=5,
        description="Simple branching tree",
    ),
    "bush": LSystemPreset(
        axiom="F",
        rules={"F": "FF+[+F-F-F]-[-F+F+F]"},
        default_angle=22.5,
        default_iterations=4,
        description="Dense bush with many branches",
    ),
    "fern": LSystemPreset(
        axiom="X",
        rules={"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"},
        default_angle=25.0,
        default_iterations=6,
        description="Fern-like branching pattern",
    ),
    "hilbert3d": LSystemPreset(
        axiom="A",
        rules={
            "A": "B-F+CFC+F-D&F^D-F+&&CFC+F+B//",
            "B": "A&F^CFB^F^D^^-F-D^|F^B|FC^F^A//",
            "C": "|D^|F^B-F+C^F^A&&FA&F^C+F+B^F^D//",
            "D": "|CFB-F+B|FA&F^A&&FB-F+B|FC//",
        },
        default_angle=90.0,
        default_iterations=2,
        description="3D Hilbert space-filling curve",
    ),
    "sierpinski": LSystemPreset(
        axiom="F-G-G",
        rules={"F": "F-G+F+G-F", "G": "GG"},
        default_angle=120.0,
        default_iterations=6,
        description="Sierpinski triangle as a continuous path",
    ),
}


def rewrite(axiom: str, rules: dict[str, str], iterations: int) -> str:
    """Apply L-system production rules for the given number of iterations."""
    current = axiom
    for _ in range(iterations):
        next_str: list[str] = []
        for ch in current:
            next_str.append(rules.get(ch, ch))
        current = "".join(next_str)
    return current


@dataclass
class _TurtleState:
    """Snapshot of turtle position and orientation."""

    position: np.ndarray
    heading: np.ndarray
    left: np.ndarray
    up: np.ndarray
    thickness: float


@dataclass
class Segment:
    """A single line segment produced by the turtle."""

    start: np.ndarray
    end: np.ndarray
    thickness: float


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix around a unit axis."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def _rotate_turtle(
    state: _TurtleState, axis: np.ndarray, angle_rad: float,
) -> None:
    """Rotate turtle orientation vectors around the given axis."""
    rot = _rotation_matrix(axis, angle_rad)
    state.heading = rot @ state.heading
    state.left = rot @ state.left
    state.up = rot @ state.up


def interpret_turtle(
    instruction_string: str,
    angle_deg: float,
    length: float,
    length_decay: float,
    thickness_decay: float,
    jitter_deg: float,
    rng: Generator,
) -> list[Segment]:
    """Interpret an L-system string with a 3D turtle.

    Turtle commands:
      F, G  - move forward, drawing a segment
      f     - move forward without drawing
      +     - yaw right (turn around up axis)
      -     - yaw left
      ^     - pitch up (turn around left axis)
      &     - pitch down
      /     - roll clockwise (turn around heading axis)
      \\    - roll counter-clockwise
      |     - turn 180 degrees
      [     - push state
      ]     - pop state
    """
    angle_rad = np.radians(angle_deg)
    jitter_rad = np.radians(jitter_deg)

    state = _TurtleState(
        position=np.array([0.0, 0.0, 0.0]),
        heading=np.array([0.0, 0.0, 1.0]),
        left=np.array([-1.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        thickness=1.0,
    )

    stack: list[_TurtleState] = []
    segments: list[Segment] = []
    depth = 0

    for ch in instruction_string:
        if ch in ("F", "G"):
            step = length * (length_decay ** depth)
            jitter = rng.uniform(-jitter_rad, jitter_rad)
            effective_angle = jitter
            if abs(effective_angle) > 1e-10:
                random_axis = rng.standard_normal(3)
                norm = np.linalg.norm(random_axis)
                if norm > 1e-10:
                    random_axis /= norm
                    _rotate_turtle(state, random_axis, effective_angle)

            start = state.position.copy()
            state.position = start + state.heading * step
            thickness = state.thickness * (thickness_decay ** depth)
            segments.append(Segment(
                start=start, end=state.position.copy(), thickness=thickness,
            ))
        elif ch == "f":
            step = length * (length_decay ** depth)
            state.position = state.position + state.heading * step
        elif ch == "+":
            _rotate_turtle(state, state.up, angle_rad)
        elif ch == "-":
            _rotate_turtle(state, state.up, -angle_rad)
        elif ch == "^":
            _rotate_turtle(state, state.left, angle_rad)
        elif ch == "&":
            _rotate_turtle(state, state.left, -angle_rad)
        elif ch == "/":
            _rotate_turtle(state, state.heading, angle_rad)
        elif ch == "\\":
            _rotate_turtle(state, state.heading, -angle_rad)
        elif ch == "|":
            _rotate_turtle(state, state.up, np.pi)
        elif ch == "[":
            stack.append(_TurtleState(
                position=state.position.copy(),
                heading=state.heading.copy(),
                left=state.left.copy(),
                up=state.up.copy(),
                thickness=state.thickness,
            ))
            depth += 1
        elif ch == "]":
            if stack:
                state = stack.pop()
                depth = max(0, depth - 1)

    return segments
