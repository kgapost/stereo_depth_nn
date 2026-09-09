"""AirSim-only helpers shared by the AirSim path (airsim_demo.py and publish_airsim.py).

Also holds ALL AirSim -> PX4 uORB translation (see the second half of this file):
publish_airsim.py turns the simulator's state into the same /fmu/out/* messages a
real PX4 flight controller emits over the uXRCE-DDS bridge, so every consumer -
and every rosbag - sees one topic set and one message type per quantity whether
the data came from AirSim or from real hardware.

This module deliberately imports NEITHER `airsim` NOR `px4_msgs`: the translation
functions duck-type the objects they are handed. That keeps the file importable on
the Jetson (where the airsim package is absent) and inside utils_drone_state.py.
"""

import os
import math
import time
import subprocess

import config

# ===== AirSim related =====
# Used by the AirSim path only (airsim_demo.py / publish_airsim.py via utils_airsim.launch_sim
# and update_settings_json): manual flight speeds, the sim binary path, and the
# settings.json template/output locations. No effect in ROS/Jetson mode.
JSON_NOISE = False
SPEED = 5.0
Z_SPEED = 5.0
TURN_SPEED = 15.0
# AirSim_Binary is a sibling of the project root (both live under
# .../AirSim/), so binaries are located relative to config.ROOT_PATH rather
# than a hardcoded absolute path.
AIRSIM_BINARY_ROOT = os.path.join(os.environ.get("AIRSIM_BINARY_ROOT", config.ROOT_PATH), 'AirSim')

ENV_SCRIPT = os.path.join(AIRSIM_BINARY_ROOT, 'AirSimNH', 'LinuxNoEditor', 'AirSimNH.sh')
# ENV_SCRIPT = os.path.join(AIRSIM_BINARY_ROOT, 'AbandonedPark', 'LinuxNoEditor', 'AbandonedPark.sh')
# ENV_SCRIPT = os.path.join(AIRSIM_BINARY_ROOT, 'Africa_Savannah', 'LinuxNoEditor', 'Africa_001.sh')
# ENV_SCRIPT = os.path.join(AIRSIM_BINARY_ROOT, 'Blocks', 'LinuxBlocks1.8.1', 'LinuxNoEditor', 'Blocks.sh')
# ENV_SCRIPT = os.path.join(AIRSIM_BINARY_ROOT, 'LandscapeMountains', 'LinuxNoEditor', 'LandscapeMountains.sh')
# ENV_SCRIPT = os.path.join(AIRSIM_BINARY_ROOT, 'ZhangJiajie', 'LinuxNoEditor', 'ZhangJiajie.sh')
# ENV_SCRIPT = os.path.join(AIRSIM_BINARY_ROOT, 'TrapCam', 'LinuxNoEditor', 'TrapCam.sh')

JSON_TEMPLATE = os.path.join(config.ROOT_PATH, 'json_templates', 'settings_template.json')
JSON_SETTINGS = os.path.join(config.ROOT_PATH, 'json_templates', 'settings.json')
# AirSim render resolution - the size AirSim is told to render and feed
# (settings.json + window ResX/ResY). Only dictates what the simulator produces;
# it has no effect on the real drone camera.
IMG_W_AIRSIM = 448
IMG_H_AIRSIM = 256

def update_settings_json(json_template, json_settings):
    """Write AirSim's settings.json from the template, injecting the IMG_*_AIRSIM
    render size, noise flag, and LiDAR parameters."""
    LIDAR_CHANNELS = 16
    LIDAR_POINTS_PS = 10000
    LIDAR_ROTATION_PS = 10
    LIDAR_HORZ_ROTATION_RES = 0.5

    if not os.path.exists(json_template):
        print(f"Settings template not found at {json_template}")
        return
    with open(json_template, 'r') as f:
        config_text = f.read()
    config_text = config_text.replace('"NOISE"', str(JSON_NOISE).lower())
    config_text = config_text.replace('"IMG_W"', str(IMG_W_AIRSIM))
    config_text = config_text.replace('"IMG_H"', str(IMG_H_AIRSIM))
    config_text = config_text.replace('"LIDAR_CHANNELS"', str(LIDAR_CHANNELS))
    config_text = config_text.replace('"LIDAR_POINTS_PS"', str(LIDAR_POINTS_PS))
    config_text = config_text.replace('"LIDAR_HORZ_ROTATION_RES"', str(LIDAR_HORZ_ROTATION_RES))
    config_text = config_text.replace('"LIDAR_ROTATION_PS"', str(LIDAR_ROTATION_PS))
    with open(json_settings, 'w') as f:
        f.write(config_text)
    print("Updated AirSim settings.json.")


def launch_sim(sim_script, show_output=False):
    """Launch the AirSim simulator binary as a detached subprocess at the
    configured render resolution. Returns the Popen handle."""
    command = [
        f'{sim_script}',
        '-windowed',
        '-nosound',
        f'-ResX={IMG_W_AIRSIM}',
        f'-ResY={IMG_H_AIRSIM}',
        f'-settings="{JSON_SETTINGS}"'
    ]
    child_env = os.environ.copy()
    if 'DISPLAY' not in child_env:
        child_env['DISPLAY'] = ':0'
    stdout = None if show_output else subprocess.DEVNULL
    stderr = None if show_output else subprocess.DEVNULL
    sim_process = subprocess.Popen(
        command,
        env=child_env,
        stdout=stdout,
        stderr=stderr,
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
    )
    print("Waiting for Unreal Engine...", end='', flush=True)
    for i in range(5):
        time.sleep(0.5)
        print('.', end='', flush=True)
    print(" started.")
    return sim_process


# =========================================================================
# ===== AirSim -> PX4 uORB state translation =====
# =========================================================================
# Why this exists
# ---------------
# The SWARMER integration expects the drone state on the standard PX4 topics
# exposed to ROS 2 through the uXRCE-DDS bridge (/fmu/out/vehicle_local_position,
# /fmu/out/vehicle_attitude, /fmu/out/vehicle_global_position,
# /fmu/out/estimator_status), NOT on a project-specific geometry_msgs/PoseStamped.
# In simulation there is no flight controller to produce them, so publish_airsim.py
# synthesises them from AirSim's own state estimate using the functions below.
# node_obstacle_detection.py then consumes exactly the same topics and message
# types in simulation and on the real drone - see utils_drone_state.py.
#
# Frame conventions - the reason this is a translation and not a copy
# -------------------------------------------------------------------
# Position: AirSim's `kinematics_estimated.position` is already NED in metres
#   (x north, y east, z DOWN), with the origin at the vehicle's PlayerStart. PX4's
#   VehicleLocalPosition x/y/z is also NED in metres, with the origin at the EKF's
#   local reference. Same convention -> a direct copy, no sign flips. Altitude is
#   -z on both sides, which is what utils_drone_state.get_altitude() relies on.
#
# Attitude: AirSim's `orientation` is the body->world rotation with the body frame
#   FRD (x forward, y right, z down) and the world frame NED - identical to what
#   PX4's VehicleAttitude.q describes. The only difference is COMPONENT ORDER:
#   AirSim exposes a Quaternionr as (x_val, y_val, z_val, w_val) while PX4 packs
#   q as [w, x, y, z]. Getting this wrong yields a rotation that looks plausible
#   but is wrong, so it is done once, here.
#
# Geodetic: PX4's VehicleGlobalPosition is lat/lon in degrees + altitude in metres
#   AMSL. AirSim can supply this two ways, and we prefer them in this order:
#     1. `state.gps_location` - the simulator's own GPS fix. Correct by
#        construction, but NaN/zero when GPS is disabled in settings.json.
#     2. Projection of the NED offset onto the home geo point (geodetic_from_ned
#        below). Always available, accurate to well under a metre at the scales a
#        drone flies over.
#
# Timestamps: PX4 stamps in MICROSECONDS (uint64). On real hardware that is time
#   since flight-controller boot; here it is wall-clock microseconds, which keeps
#   it monotonic and comparable with the ROS header stamps in the same rosbag.

# WGS-84 semi-major axis; the sphere approximation below is accurate to ~0.5% of
# the offset, i.e. centimetres over the hundreds of metres a flight covers.
EARTH_RADIUS_M = 6378137.0

# AirSim's built-in geodetic origin (Microsoft's Redmond campus) - what
# getHomeGeoPoint() returns when settings.json has no "OriginGeopoint" entry, as
# json_templates/settings_template.json currently does not. Only used as a
# last-resort fallback if the home point cannot be read from the simulator at all;
# add an "OriginGeopoint" to the template to fly the sim over real coordinates.
AIRSIM_DEFAULT_HOME_GEO = (47.641468, -122.140165, 122.0)


def px4_timestamp_us():
    """Current time in microseconds, as PX4 stamps its uORB messages (uint64)."""
    return int(time.time() * 1e6)


def _finite(value):
    """True when `value` is a usable real number. AirSim reports NaN (and
    occasionally an exact 0.0 triple) for GPS when the sensor is disabled in
    settings.json, which must not be forwarded as a valid fix."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def geodetic_from_ned(home_geo, north_m, east_m, down_m):
    """Project a local NED offset (metres) onto a geodetic home point.

    `home_geo` is (lat_deg, lon_deg, alt_m_amsl). Returns the same triple for the
    offset position. Uses the equirectangular (flat-earth) approximation around
    the home latitude - the standard choice for local navigation frames, and what
    PX4's own map_projection does over these distances."""
    home_lat, home_lon, home_alt = home_geo
    lat = home_lat + math.degrees(north_m / EARTH_RADIUS_M)
    # Longitude degrees shrink with latitude; guard the cosine at the poles so a
    # degenerate home point cannot produce a division by zero.
    cos_lat = max(math.cos(math.radians(home_lat)), 1e-9)
    lon = home_lon + math.degrees(east_m / (EARTH_RADIUS_M * cos_lat))
    alt = home_alt - down_m          # NED down is positive; altitude is up
    return lat, lon, alt


def quat_wxyz_to_yaw(qw, qx, qy, qz):
    """Yaw (rad, [-pi, pi]) of a body(FRD)->NED quaternion given as w,x,y,z.
    This is PX4's `heading` field in VehicleLocalPosition."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def airsim_quaternion_to_px4(orientation):
    """AirSim Quaternionr (x_val, y_val, z_val, w_val) -> PX4 q order [w, x, y, z].

    Both describe the same body(FRD)->world(NED) rotation; only the packing order
    differs. Kept as its own function because this reorder is the single easiest
    thing in the whole bridge to get silently wrong."""
    return [float(orientation.w_val), float(orientation.x_val),
            float(orientation.y_val), float(orientation.z_val)]


def airsim_pose_to_px4_frame(pose):
    """AirSim Pose (as returned by simGetCameraInfo / kinematics_estimated) ->
    (position_ned, quat_xyzw).

    `position_ned` is (x, y, z) in NED metres and `quat_xyzw` is ordered for
    scipy's Rotation.from_quat(). This is the in-process equivalent of what
    airsim_state_to_px4() produces for the ROS path, so utils_drone_state.py
    yields identical numbers whether the pose arrived over /fmu/out/* or straight
    off the AirSim API."""
    position = (float(pose.position.x_val), float(pose.position.y_val),
                float(pose.position.z_val))
    q = pose.orientation
    quat_xyzw = (float(q.x_val), float(q.y_val), float(q.z_val), float(q.w_val))
    return position, quat_xyzw


def airsim_state_to_px4(state, home_geo=None, timestamp_us=None):
    """Translate one AirSim MultirotorState into the PX4 uORB quantities.

    Returns a plain dict - no ROS types, no px4_msgs import - so the conversion is
    testable on its own and reusable by the non-ROS AirSim path. The fill_*
    helpers below turn this dict into actual px4_msgs messages.

    `home_geo` is the (lat, lon, alt) origin from client.getHomeGeoPoint(); when
    omitted, AIRSIM_DEFAULT_HOME_GEO is assumed."""
    if home_geo is None:
        home_geo = AIRSIM_DEFAULT_HOME_GEO
    if timestamp_us is None:
        timestamp_us = px4_timestamp_us()

    kin = state.kinematics_estimated
    x, y, z = (float(kin.position.x_val), float(kin.position.y_val),
               float(kin.position.z_val))
    q = airsim_quaternion_to_px4(kin.orientation)

    # Velocity / acceleration are NED on both sides, like position.
    vel = getattr(kin, 'linear_velocity', None)
    vx, vy, vz = ((float(vel.x_val), float(vel.y_val), float(vel.z_val))
                  if vel is not None else (0.0, 0.0, 0.0))
    acc = getattr(kin, 'linear_acceleration', None)
    ax, ay, az = ((float(acc.x_val), float(acc.y_val), float(acc.z_val))
                  if acc is not None else (0.0, 0.0, 0.0))

    # Geodetic position: prefer the simulator's own GPS fix, fall back to
    # projecting the NED offset onto the home point (see the notes above).
    gps = getattr(state, 'gps_location', None)
    if (gps is not None and _finite(getattr(gps, 'latitude', None))
            and _finite(getattr(gps, 'longitude', None))
            and _finite(getattr(gps, 'altitude', None))
            and not (gps.latitude == 0.0 and gps.longitude == 0.0)):
        lat, lon, alt = float(gps.latitude), float(gps.longitude), float(gps.altitude)
        geo_source = 'airsim_gps'
    else:
        lat, lon, alt = geodetic_from_ned(home_geo, x, y, z)
        geo_source = 'projected_from_ned'

    return {
        'timestamp_us': timestamp_us,
        # VehicleLocalPosition
        'x': x, 'y': y, 'z': z,
        'vx': vx, 'vy': vy, 'vz': vz,
        'ax': ax, 'ay': ay, 'az': az,
        'heading': quat_wxyz_to_yaw(*q),
        # VehicleAttitude
        'q': q,                       # [w, x, y, z], body FRD -> NED
        'quat_xyzw': (q[1], q[2], q[3], q[0]),   # scipy / PoseStamped order
        # VehicleGlobalPosition
        'lat': lat, 'lon': lon, 'alt': alt,
        'geo_source': geo_source,
        'home_geo': tuple(home_geo),
        # Simulated sensor quality. AirSim's state estimate is ground truth, so
        # these advertise a converged, healthy EKF rather than pretending to a
        # realistic error budget - a consumer gating on EKF health must not stall
        # forever in simulation.
        'eph': 0.3, 'epv': 0.5,
        'healthy': True,
    }


def _set_if_present(msg, field, value):
    """Assign `field` on `msg` only if the message definition has it.

    The uORB definitions drift between PX4 releases (VehicleGlobalPosition alone
    has gained and lost fields across v1.13/v1.14/main), and px4_msgs is built
    from whichever release the integrator's workspace pinned. Setting fields
    defensively means this bridge keeps working against a px4_msgs we did not
    build, instead of dying on an AttributeError for one cosmetic field."""
    if hasattr(msg, field):
        setattr(msg, field, value)
        return True
    return False


def fill_vehicle_local_position(msg, px4_state):
    """Populate a px4_msgs VehicleLocalPosition from airsim_state_to_px4()."""
    ts = px4_state['timestamp_us']
    _set_if_present(msg, 'timestamp', ts)
    _set_if_present(msg, 'timestamp_sample', ts)
    # Validity flags: utils_drone_state.px4_local_position_cb drops any sample
    # with xy_valid/z_valid False, so these must be set or the pose never moves.
    _set_if_present(msg, 'xy_valid', True)
    _set_if_present(msg, 'z_valid', True)
    _set_if_present(msg, 'v_xy_valid', True)
    _set_if_present(msg, 'v_z_valid', True)
    _set_if_present(msg, 'x', float(px4_state['x']))
    _set_if_present(msg, 'y', float(px4_state['y']))
    _set_if_present(msg, 'z', float(px4_state['z']))
    _set_if_present(msg, 'vx', float(px4_state['vx']))
    _set_if_present(msg, 'vy', float(px4_state['vy']))
    _set_if_present(msg, 'vz', float(px4_state['vz']))
    _set_if_present(msg, 'ax', float(px4_state['ax']))
    _set_if_present(msg, 'ay', float(px4_state['ay']))
    _set_if_present(msg, 'az', float(px4_state['az']))
    _set_if_present(msg, 'heading', float(px4_state['heading']))
    # Geodetic reference of the local frame, so a consumer can do its own
    # local->global conversion exactly as it would against a real EKF2.
    home_lat, home_lon, home_alt = px4_state['home_geo']
    _set_if_present(msg, 'xy_global', True)
    _set_if_present(msg, 'z_global', True)
    _set_if_present(msg, 'ref_timestamp', ts)
    _set_if_present(msg, 'ref_lat', float(home_lat))
    _set_if_present(msg, 'ref_lon', float(home_lon))
    _set_if_present(msg, 'ref_alt', float(home_alt))
    _set_if_present(msg, 'eph', float(px4_state['eph']))
    _set_if_present(msg, 'epv', float(px4_state['epv']))
    return msg


def fill_vehicle_attitude(msg, px4_state):
    """Populate a px4_msgs VehicleAttitude from airsim_state_to_px4().

    msg.q is a fixed-size float32[4] array; px4_msgs exposes it as a numpy array,
    so assign element-wise rather than rebinding it to a Python list."""
    ts = px4_state['timestamp_us']
    _set_if_present(msg, 'timestamp', ts)
    _set_if_present(msg, 'timestamp_sample', ts)
    q = px4_state['q']
    try:
        for i in range(4):
            msg.q[i] = float(q[i])
    except (TypeError, AttributeError):
        msg.q = [float(v) for v in q]
    return msg


def fill_vehicle_global_position(msg, px4_state):
    """Populate a px4_msgs VehicleGlobalPosition from airsim_state_to_px4()."""
    ts = px4_state['timestamp_us']
    _set_if_present(msg, 'timestamp', ts)
    _set_if_present(msg, 'timestamp_sample', ts)
    _set_if_present(msg, 'lat', float(px4_state['lat']))
    _set_if_present(msg, 'lon', float(px4_state['lon']))
    _set_if_present(msg, 'alt', float(px4_state['alt']))
    _set_if_present(msg, 'alt_ellipsoid', float(px4_state['alt']))
    # utils_drone_state.px4_global_position_cb gates 'valid' on these two.
    _set_if_present(msg, 'lat_lon_valid', True)
    _set_if_present(msg, 'alt_valid', True)
    _set_if_present(msg, 'eph', float(px4_state['eph']))
    _set_if_present(msg, 'epv', float(px4_state['epv']))
    _set_if_present(msg, 'dead_reckoning', False)
    _set_if_present(msg, 'terrain_alt_valid', False)
    return msg


def fill_estimator_status(msg, px4_state):
    """Populate a px4_msgs EstimatorStatus from airsim_state_to_px4().

    Advertises a converged filter with no faults - see the note on 'healthy' in
    airsim_state_to_px4()."""
    ts = px4_state['timestamp_us']
    _set_if_present(msg, 'timestamp', ts)
    _set_if_present(msg, 'timestamp_sample', ts)
    _set_if_present(msg, 'filter_fault_flags', 0)
    _set_if_present(msg, 'pos_horiz_accuracy', float(px4_state['eph']))
    _set_if_present(msg, 'pos_vert_accuracy', float(px4_state['epv']))
    for flag in ('pre_flt_fail_innov_heading', 'pre_flt_fail_innov_vel_horiz',
                 'pre_flt_fail_innov_vel_vert', 'pre_flt_fail_innov_height'):
        _set_if_present(msg, flag, False)
    return msg
