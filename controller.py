import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

def refine_points(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    points: numpy array of shape (N, 2)
    iterations: number of times to replace points with midpoints
    
    Returns an array of shape (N-iterations, 2)
    """
    pts = points
    for _ in range(iterations):
        new_pts = []
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            mid = (p0 + p1) / 2.0
            new_pts.append(p0)
            new_pts.append(mid)
        new_pts.append(pts[-1])  # include last point
        pts = np.array(new_pts)
    return pts

# Low-level PID for steering + speed
class PID:
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.output_limits = output_limits

    def update(self, error, dt=0.1):   # simulator time_step is 0.1s
        # integrate
        self.integral += error * dt

        # derivative
        d = (error - self.prev_error) / dt
        self.prev_error = error

        # PID output
        out = self.kp * error + self.ki * self.integral + self.kd * d

        if self.output_limits is not None:
            out = np.clip(out, self.output_limits[0], self.output_limits[1])

        return out


# Instantiate low-level PIDs (same objects reused every call)
steer_pid = PID(kp=4.0, ki=0.0, kd=0.2, output_limits=(-2.0, 2.0))   # rad/s
vel_pid   = PID(kp=1.5, ki=0.1, kd=0.0, output_limits=(-5.0, 5.0))   # m/s²


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
BASE_SPEED = 35.0      # target speed on straights (m/s)
MIN_SPEED  = 8.0       # do not crawl slower than this
TURN_SLOWDOWN = 4.0    # speed reduction factor from curvature


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
    ref = refine_points(ref)
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
        base_speed = 12.0
    elif curvature > 0.07:        # medium turn
        lookahead = 4
        base_speed = 20.0
    else:                         # gentle / straight
        lookahead = 7
        base_speed = BASE_SPEED

    look_idx = (idx + lookahead) % N
    target = ref[look_idx]

    # ---- 4. Compute desired steering ANGLE ----
    dx = target[0] - sx
    dy = target[1] - sy
    target_heading = np.arctan2(dy, dx)

    heading_error = target_heading - phi
    # wrap to [-pi, pi] FIRST
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    # no pre-clipping here; just let steering saturation & PID handle it

    # Proportional steering from heading error
    steer_gain = 1.2 if curvature < 0.07 else 1.8
    if curvature > 0.15:
        steer_gain = 2.2
    desired_steer = steer_gain * heading_error

    # Clip desired steering to physical +/- max steering angle
    desired_steer = np.clip(desired_steer, parameters[1], parameters[4])

    # ---- 5. Desired speed from curvature (no crazy scaling) ----
    # Smooth slowdown: divide BASE_SPEED by (1 + TURN_SLOWDOWN * curvature)
    desired_speed = base_speed / (1.0 + TURN_SLOWDOWN * curvature)
    desired_speed = np.clip(desired_speed, MIN_SPEED, base_speed)
    # print("desired_speed:", desired_speed, "desired_steer:", desired_steer)

    # ---- 6. Send to low-level controller ----
    return lower_controller(state, np.array([desired_steer, desired_speed]), parameters)
