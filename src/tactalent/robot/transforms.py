"""Small rigid-transform helpers using scalar-first quaternions."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def vector(values: Sequence[float], size: int) -> np.ndarray:
    """Validate and return a finite vector.

    Args:
        values: Numeric input values.
        size: Required number of elements.

    Returns:
        A copied float64 array.

    Raises:
        ValueError: If the input has the wrong size or nonfinite values.
    """
    result = np.asarray(values, dtype=float).reshape(-1).copy()
    if result.size != size or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite values")
    return result


def quaternion(values: Sequence[float]) -> np.ndarray:
    """Return a normalized quaternion in wxyz order."""
    result = vector(values, 4)
    norm = np.linalg.norm(result)
    if norm < 1e-12:
        raise ValueError("Quaternion must have nonzero norm")
    return result / norm


def multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    """Compose rotations, applying right before left."""
    w, x, y, z = quaternion(left)
    a, b, c, d = quaternion(right)
    return quaternion(
        [
            w * a - x * b - y * c - z * d,
            w * b + x * a + y * d - z * c,
            w * c - x * d + y * a + z * b,
            w * d + x * c - y * b + z * a,
        ]
    )


def rotation_matrix(values: Sequence[float]) -> np.ndarray:
    """Convert a wxyz quaternion to a rotation matrix."""
    w, x, y, z = quaternion(values)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def compose(
    parent_p: Sequence[float],
    parent_q: Sequence[float],
    local_p: Sequence[float],
    local_q: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a local pose into its parent coordinate system."""
    return (
        vector(parent_p, 3) + rotation_matrix(parent_q) @ vector(local_p, 3),
        multiply(parent_q, local_q),
    )


def euler_degrees(values: Sequence[float]) -> np.ndarray:
    """Build an XYZ Euler increment in degrees (Rz @ Ry @ Rx)."""
    rx, ry, rz = np.radians(vector(values, 3)) / 2
    qx = [math.cos(rx), math.sin(rx), 0, 0]
    qy = [math.cos(ry), 0, math.sin(ry), 0]
    qz = [math.cos(rz), 0, 0, math.sin(rz)]
    return multiply(qz, multiply(qy, qx))


def move_tcp(
    position: Sequence[float],
    orientation: Sequence[float],
    delta: Sequence[float],
    frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate a TCP in base axes or its current local axes."""
    p, q, d = vector(position, 3), quaternion(orientation), vector(delta, 3)
    if frame == "tcp":
        d = rotation_matrix(q) @ d
    elif frame != "base":
        raise ValueError("Movement frame must be base or tcp")
    return p + d, q


def rotate_tcp(
    position: Sequence[float],
    orientation: Sequence[float],
    angles: Sequence[float],
    frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate about the TCP itself, preserving its base position."""
    p, q, increment = (
        vector(position, 3),
        quaternion(orientation),
        euler_degrees(angles),
    )
    if frame == "base":
        return p, multiply(increment, q)
    if frame == "tcp":
        return p, multiply(q, increment)
    raise ValueError("Rotation frame must be base or tcp")


def orientation_error(actual: Sequence[float], target: Sequence[float]) -> float:
    """Return quaternion angular distance in radians, ignoring quaternion sign."""
    dot = float(np.dot(quaternion(actual), quaternion(target)))
    return 2 * math.acos(float(np.clip(abs(dot), 0, 1)))


def point_tcp(
    position: Sequence[float],
    tilt_degrees: float,
    frame: str = "base",
    reference_orientation: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Point the approach axis, choosing the closest parallel-jaw orientation."""
    if (
        frame != "base"
        or not math.isfinite(tilt_degrees)
        or tilt_degrees not in (0, 15, 30, 45, 60, 75, 90)
    ):
        raise ValueError("Point requires a supported base-frame tilt preset")
    angle = math.radians(180 - tilt_degrees) / 2
    target = quaternion([math.cos(angle), 0, math.sin(angle), 0])
    if reference_orientation is not None:
        # A half turn around tool Z preserves approach and the unoriented jaw line.
        alternate = multiply(target, [0, 0, 0, 1])
        if orientation_error(reference_orientation, alternate) < orientation_error(
            reference_orientation, target
        ):
            target = alternate
    return vector(position, 3), target
