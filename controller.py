import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

def smooth_gain(curv, k_min, k_max, c):
    """
    Smooth exponential decay:
    - curv = curvature (0 → straight, 0.2 → sharp turn)
    - k_min = gain used in sharp turns
    - k_max = gain used on straights
    - c = how fast the gain transitions
    """
    return k_min + (k_max - k_min) * np.exp(-c * curv)


class PID:
    def __init__(self, kp, ki, kd, output_limits=None, derivative_alpha=0.25):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.output_limits = output_limits

        # Derivative smoothing
        self.derivative_alpha = derivative_alpha
        self.filtered_derivative = 0.0
        self.first_update = True

    def update(self, error, dt=0.1):
        # Integral term
        self.integral += error * dt

        # Raw derivative
        raw_derivative = (error - self.prev_error) / dt

        # EMA low-pass filter for derivative
        if self.first_update:
            # initialize on first step
            self.filtered_derivative = raw_derivative
            self.first_update = False
        else:
            a = self.derivative_alpha
            self.filtered_derivative = (
                a * raw_derivative + (1 - a) * self.filtered_derivative
            )

        # Save for next iteration
        self.prev_error = error

        # PID output using filtered derivative
        out = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * self.filtered_derivative
        )

        # Output saturation
        if self.output_limits is not None:
            out = np.clip(out, self.output_limits[0], self.output_limits[1])

        return out



# Instantiate low-level PIDs (same objects reused every call)
# steer_pid = PID(kp=4.0, ki=0.0, kd=0.2, output_limits=(-3.0, 3.0))   # rad/s
steer_pid = PID(kp=2.4, ki=0.0, kd=0.55, output_limits=(-2.0, 2.0), derivative_alpha=0.20)
vel_pid   = PID(kp=1.5, ki=0.1, kd=0.0, output_limits=(-10.0, 10.0))   # m/s²


# LOW-LEVEL CONTROLLER:
# Takes desired steering *angle* & desired speed
# Produces steering *rate* (v_delta) & acceleration
def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike):
    """
    state   = [sx, sy, delta, v, phi]
    desired = [desired_steering_angle , desired_speed]
    """
    _, _, delta, v, phi = state
    desired_steer = desired[0]
    desired_speed = desired[1]

    # --- Steering rate control ---
    steer_error = desired_steer - delta
    v_delta = steer_pid.update(steer_error)

    # Clip to physical steering-rate limits
    v_delta = np.clip(v_delta, parameters[7], parameters[9])

    # --- Velocity / acceleration control ---
    vel_error = desired_speed - v
    accel = vel_pid.update(vel_error)

    # Clip to physical acceleration limits
    accel = np.clip(accel, parameters[8], parameters[10])

    return np.array([v_delta, accel])


# GLOBAL TUNING CONSTANTS (used only as defaults)
BASE_SPEED = 170.0      # target speed on straights (m/s)
MIN_SPEED  = 10.0       # do not crawl slower than this
TURN_SLOWDOWN = 5.5    # speed reduction factor from curvature


def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    """
    High-level controller:
      - picks a lookahead target on centerline
      - computes desired steering angle
      - computes desired speed from curvature
      - then calls lower_controller
    """
    sx, sy, delta, v, phi = state

    # ---- 1. Find closest point on reference path ----
    ref = racetrack.centerline
    pos = np.array([sx, sy])
    dists = np.linalg.norm(ref - pos, axis=1)
    idx = np.argmin(dists)
    N = len(ref)

    # ---- 2. Proper curvature estimate (angle change per distance) ----
    prev = ref[(idx - 2) % N]
    nextp = ref[(idx + 2) % N]

    # headings of segments
    heading_prev = np.arctan2(prev[1] - sy, prev[0] - sx)
    heading_next = np.arctan2(nextp[1] - sy, nextp[0] - sx)
    dtheta = np.arctan2(np.sin(heading_next - heading_prev),
                        np.cos(heading_next - heading_prev))
    ds = np.linalg.norm(nextp - prev)
    curvature = abs(dtheta) / max(ds, 1e-3)   # units ~ 1/m, typically 0–0.3
    # print("curvature:", curvature)

    # ---- 3. Choose lookahead & base_speed based on curvature ----
    if curvature > 0.15:          # very tight / hairpin
        lookahead = 2
        base_speed = 46.0
    elif curvature > 0.07:        # medium turn
        lookahead = 4
        base_speed = 98.0
    else:                         # gentle / straight
        lookahead = 7
        base_speed = BASE_SPEED

    look_idx = (idx + lookahead) % N
    target = ref[look_idx]

    # steer_pid.kp = smooth_gain(curvature, 1.8, 4.5, 25)
    # steer_pid.ki = smooth_gain(curvature, 0.00, 0.05, 25)
    # steer_pid.kd = smooth_gain(curvature, 0.45, 0.10, 25)

    # ---- 4. Compute desired steering ANGLE ----
    dx = target[0] - sx
    dy = target[1] - sy
    target_heading = np.arctan2(dy, dx)

    heading_error = target_heading - phi
    # wrap to [-pi, pi] FIRST
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    # no pre-clipping here; just let steering saturation & PID handle it

    # Proportional steering from heading error
    if curvature > 0.15:
        steer_gain = 2.6
    elif curvature > 0.07:
        steer_gain = 2.0
    else:
        steer_gain = 1.4

    # steer_gain = smooth_gain(curvature, 1.2, 2.8, 18)

    # desired_steer = steer_gain * heading_error
    damp = 1.0 / (1.0 + 2.5 * abs(heading_error))
    desired_steer = steer_gain * heading_error * damp

    # Clip desired steering to physical +/- max steering angle
    desired_steer = np.clip(desired_steer, parameters[1], parameters[4])

    # ---- 5. Desired speed from curvature (no crazy scaling) ----
    # Smooth slowdown: divide BASE_SPEED by (1 + TURN_SLOWDOWN * curvature)
    desired_speed = base_speed / (1.0 + TURN_SLOWDOWN * curvature)
    desired_speed = np.clip(desired_speed, MIN_SPEED, base_speed)
    # print("desired_speed:", desired_speed, "desired_steer:", desired_steer)

    # ---- 6. Send to low-level controller ----
    return lower_controller(state, np.array([desired_steer, desired_speed]), parameters)
