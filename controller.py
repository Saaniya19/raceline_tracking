import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack


# Low-level PID for steering + speed
class PID:
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.output_limits = output_limits

    def update(self, error, dt=0.02):
        # integrate
        self.integral += error * dt

        # derivative
        d = (error - self.prev_error) / dt

        # PID output
        out = self.kp * error + self.ki * self.integral + self.kd * d
        self.prev_error = error

        if self.output_limits is not None:
            out = np.clip(out, self.output_limits[0], self.output_limits[1])

        return out


# Instantiate low-level PIDs
steer_pid = PID(kp=4.0, ki=0.0, kd=0.2, output_limits=(-2.0, 2.0))     # rad/s
vel_pid   = PID(kp=1.5, ki=0.1, kd=0.0, output_limits=(-5.0, 5.0))     # m/s²


# LOW-LEVEL CONTROLLER
# Takes desired steering *angle* & desired speed
# Produces steering *rate* (v_delta) & acceleration
def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike):
    """
    state = [sx, sy, φ, v, δ]
    desired = [desired_steering_angle , desired_speed]
    """

    _, _, delta, v, phi = state
    desired_steer = desired[0]
    desired_speed = desired[1]

    # Steering control → produce steering rate
    steer_error = desired_steer - delta
    v_delta = steer_pid.update(steer_error)

    # Clip to physical limits
    v_delta = np.clip(v_delta, parameters[7], parameters[9])  # steering vel limits

    # Velocity control → produce acceleration
    vel_error = desired_speed - v
    accel = vel_pid.update(vel_error)

    # Clip to physical limits
    accel = np.clip(accel, parameters[8], parameters[10])  # acceleration limits

    return np.array([v_delta, accel])



# HIGH-LEVEL CONTROLLER
# Computes desired steering ANGLE and desired speed
LOOKAHEAD = 10          # number of points ahead on the raceline
BASE_SPEED = 1000        # m/s target on straights
TURN_SLOWDOWN = 990      # how much to reduce speed in curves


def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    """
    Produces desired steering angle and desired speed.
    Then passes them to the low-level controller.
    """

    sx, sy, delta, v, phi = state

    # ---- 1. Find closest point on reference path ----
    ref = racetrack.centerline
    pos = np.array([sx, sy])
    dists = np.linalg.norm(ref - pos, axis=1)
    idx = np.argmin(dists)

    # ---- 2. Select lookahead target ----
    N = len(ref)
    look_idx = (idx + LOOKAHEAD) % N
    target = ref[look_idx]

    # ---- 3. Compute desired steering ANGLE ----
    dx = target[0] - sx
    dy = target[1] - sy
    target_heading = np.arctan2(dy, dx)

    heading_error = target_heading - phi
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # wrap

    # Desired steering ANGLE is proportional to heading error
    desired_steer = heading_error * 1.2
    desired_steer = np.clip(desired_steer, parameters[1], parameters[4])

    # ---- 4. Compute desired SPEED based on curvature ----
    prev = ref[(idx - 2) % N]
    nextp = ref[(idx + 2) % N]

    angle_prev = np.arctan2(prev[1] - sy, prev[0] - sx)
    angle_next = np.arctan2(nextp[1] - sy, nextp[0] - sx)
    curvature = abs(np.arctan2(np.sin(angle_next - angle_prev),
                               np.cos(angle_next - angle_prev)))

    desired_speed = BASE_SPEED / (1 + TURN_SLOWDOWN * curvature)
    desired_speed = np.clip(desired_speed, 5, BASE_SPEED)

    # ---- 5. LOWER-LEVEL CONTROLLER ----
    return lower_controller(state, np.array([desired_steer, desired_speed]), parameters)
