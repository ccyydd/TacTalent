"""Native RGB storage and recorded physical clocks survive preparation."""

import cv2
import numpy as np
import pytest

from tactalent.benchmarks.openvitac.recordings import decode, validate_recorded_arrays


def test_numeric_sensor_rgb_is_not_swapped_after_native_jpeg_encoding():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:] = [220, 30, 10]
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    decoded = decode([encoded.tobytes()])[0]
    np.testing.assert_allclose(decoded[16, 16], image[16, 16], atol=3)


def test_missing_recording_intervals_preserve_the_original_step_clock():
    step = np.arange(40, dtype=np.int64) * 2
    step[2:] += 4
    ee = np.zeros((len(step), 7))
    ee[:, 3] = 1
    joint = np.zeros((len(step), 9))
    joint[:, 7:] = 0.02
    restored, _, _ = validate_recorded_arrays(step, ee, joint)
    np.testing.assert_array_equal(restored, step)
    invalid = step.copy()
    invalid[2] = invalid[1]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_recorded_arrays(invalid, ee, joint)
