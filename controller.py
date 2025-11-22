import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

# ================================
# Hyperparameters (tunable)
# ================================
LOOKAHEAD_DISTANCE = 0.02       # meters (Pure Pursuit lookahead)
KP_V = 25                     # P-gain for velocity
KD_V = 0.1                     # D-gain for velocity

KP_LAT = 20.0                   # Lateral error proportional gain
KD_LAT = 0.35                  # Heading error derivative gain

# ================
# CONTROLLER C1
# ================
# Longitudinal (velocity) controller
def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    state = [sx, sy, φ, v, δ]
    desired = [desired_steer (ignored here), desired_velocity]
    """
    print("lower")
    _, _, _, v, _ = state
    desired_speed = desired[1]

    # PD velocity controller
    error = desired_speed - v
    d_error = -v   # approximate dv/dt since model doesn't give acceleration

    a = KP_V * error + KD_V * d_error

    # Clamp acceleration
    a = np.clip(
        a,
        parameters[8],   # min acceleration
        parameters[10]   # max acceleration
    )

    # Output: steering rate ignored here → 0
    return np.array([0.0, a])


# ================
# CONTROLLER C2
# ================
# Full car controller combining velocity + steering
def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    Main controller used by the simulator.
    Returns:
        [v_delta, acceleration]
    """
    print("controller")

    sx, sy, phi, v, delta = state

    # 1. Compute nearest point on raceline
    ref = racetrack.centerline  # Nx2 array
    pos = np.array([sx, sy])
    dists = np.linalg.norm(ref - pos, axis=1)
    idx = np.argmin(dists)

    # 2. Lookahead point for Pure Pursuit
    N = len(ref)
    lookahead_idx = (idx + int(LOOKAHEAD_DISTANCE * 2)) % N
    target = ref[lookahead_idx]

    # 3. Lateral tracking: Pure Pursuit + heading error correction
    dx = target[0] - sx
    dy = target[1] - sy

    # angle to target
    angle_to_target = np.arctan2(dy, dx)

    # heading error
    heading_error = angle_to_target - phi
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # wrap

    # lateral controller (PD)
    steer_cmd = KP_LAT * heading_error + KD_LAT * (0 - delta)

    # clamp steering
    steer_cmd = np.clip(
        steer_cmd,
        parameters[1],   # min steering angle
        parameters[4]    # max steering angle
    )

    # steering rate approximation
    v_delta = (steer_cmd - delta) * 5.0   # drives steering toward target

    # 4. Desired speed from curvature
    # Compute local curvature k = |dtheta/ds|
    prev = ref[(idx - 2) % N]
    nextp = ref[(idx + 2) % N]
    heading_prev = np.arctan2(prev[1] - sy, prev[0] - sx)
    heading_next = np.arctan2(nextp[1] - sy, nextp[0] - sx)

    dtheta = np.arctan2(np.sin(heading_next - heading_prev),
                        np.cos(heading_next - heading_prev))
    curvature = abs(dtheta) + 1e-4  # avoid divide by zero

    # speed profile: slow on tight turns, fast on straights
    desired_speed = np.clip(8.0 / curvature, 3.0, 30.0)

    # 5. Call longitudinal controller
    accel_cmd = lower_controller(
        state,
        np.array([steer_cmd, desired_speed]),
        parameters,
    )[1]

    return np.array([v_delta, accel_cmd])
