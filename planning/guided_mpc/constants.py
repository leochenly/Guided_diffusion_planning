from __future__ import annotations

import numpy as np

FINISH_Y = 0.35
CENTER_X = 0.5
DEFAULT_FIXED_QUAT = np.array([0, 1, 0, 0], dtype=np.float32)

DEFAULT_OBSTACLES_XY = np.array(
    [
        [0.5, -0.1],
        [0.425, 0.08],
        [0.575, 0.08],
        [0.35, 0.26],
        [0.5, 0.26],
        [0.65, 0.26],
    ],
    dtype=np.float32,
)
