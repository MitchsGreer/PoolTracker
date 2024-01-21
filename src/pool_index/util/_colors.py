"""Colors of pool balls."""
from enum import Enum


class BallColors(Enum):
    """Colors in pool balls in RGB values."""

    YELLOW = (255, 255, 0)
    BLUE = (0, 0, 150)
    RED = (255, 0, 0)
    PURPLE = (115, 0, 125)
    ORANGE = (255, 125, 0)
    GREEN = (0, 175, 0)
    MAROON = (125, 0, 0)
    BLACK = (0, 0, 0)
    PINK = (255, 125, 175)
    WHITE = (255, 255, 255)
