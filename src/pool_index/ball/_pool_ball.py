"""Pool ball class."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PoolBall:
    bbox: Tuple[  # Bounding box of the pool ball in center_x, center_y, width, hieght format.
        int, int, int, int
    ]
    color: Tuple[int, int, int]  # The color of the ball in RGB values.
    number: int  # The number displayed on the ball.
