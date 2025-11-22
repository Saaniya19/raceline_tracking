import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

# reach a constant desired speed with one controller (C1)
def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    # [steer angle, velocity]
    assert(desired.shape == (2,))

    return np.array([0, 100]).T

# fix steering angle to stay on the reference line with the other controller (C2)
def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    return np.array([0, 100]).T