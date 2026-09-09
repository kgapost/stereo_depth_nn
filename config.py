import os, threading

# === Force RMW Implementation for ROS 2 (must be set before rclpy import) ===
# This ensures all nodes use the same middleware,
# preventing "sequence size exceeds buffer"
# and improving large message handling (images, pointclouds).
# Options:
#   - rmw_fastrtps_cpp   :  Default, uses Fast DDS
#                           can have buffer issues with large msgs.
#   - rmw_cyclonedds_cpp :  Recommended; handles large data well, robust, open-source.
#   - rmw_connextdds     :  Commercial, robust but not free for commercial use.
#   - rmw_zenoh_cpp      :  New lightweight non-DDS, still maturing.

if os.environ.get("RMW_IMPLEMENTATION") is None:
    os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    print(f"[config] Set RMW_IMPLEMENTATION to {os.environ['RMW_IMPLEMENTATION']}")
else:
    print(f"[config] RMW_IMPLEMENTATION already set to {os.environ['RMW_IMPLEMENTATION']}")

# === Force ROS_DOMAIN_ID (must be set before rclpy import) ===
# A mismatch here means the dev PC (publish_airsim.py) and the Jetson
# (node_obstacle_detection.py) are on different DDS domains and never discover each
# other at all - not just slow/lossy data. Driving it from this shared config
# (instead of each machine's own shell/Dockerfile env) keeps both sides in sync.
# Override per-machine without editing this file via the env var:
#   export ROS_DOMAIN_ID=42
# (This one keeps its standard ROS name - no OD_ prefix - because rclpy and every
# other ROS tool read it directly.)
ROS_DOMAIN_ID = 42
if os.environ.get("ROS_DOMAIN_ID") is None:
    os.environ["ROS_DOMAIN_ID"] = str(ROS_DOMAIN_ID)
    print(f"[config] Set ROS_DOMAIN_ID to {os.environ['ROS_DOMAIN_ID']}")
else:
    print(f"[config] ROS_DOMAIN_ID already set to {os.environ['ROS_DOMAIN_ID']}")

import cv2


# =========================================================================
# ===== Environment-variable overrides =====
# =========================================================================
# The field-tunable parameters in this file are declared as
#
#   NAME = _env_<type>('NAME', <default>)
#
# which reads the environment variable OD_NAME and falls back to <default> - the
# literal written right here - when it is unset or empty. So config.py stays the
# single source of truth for every default, while a field session can change any
# of them with `export OD_NAME=...` before launching: no edit to a file that is
# mounted read-only into a container on the Jetson, and no rebuild.
#
# Only the ENVIRONMENT VARIABLE carries the OD_ prefix. The config parameter keeps
# its own name, so every `config.NAME` reference across the codebase is unchanged.
#
# tools/env_vars.sh lists every overridable parameter with its default.
#
# Parameters NOT declared through an _env_* helper are deliberately fixed - either
# design choices (see the "Constants" section at the bottom of this file) or debug
# switches that must stay off in the field (see the Debug section below).

_ENV_PREFIX = 'OD_'

# Env var names that predate the OD_ prefix. Still honoured, because tools/*.sh,
# the Dockerfile and the README already use them; OD_<NAME> wins when both are set.
_ENV_LEGACY = {
    'MODE':               'OBSTACLE_DETECTION_MODE',
    'USE_DIRECT_CAMERA':  'OBSTACLE_DETECTION_USE_DIRECT_CAMERA',
    'AIRSIM_DEMO_VIZ':    'OBSTACLE_DETECTION_AIRSIM_DEMO_VIZ',
    'OCTOMAP_SHARE_MODE': 'OBSTACLE_DETECTION_OCTOMAP_SHARE_MODE',
}

# (param_name, env_var, default, parsed_value) for every override actually applied.
# Printed by _env_report() at the end of this file.
_ENV_OVERRIDES = []


def _env_lookup(name):
    """(env_var, raw_string) for config parameter `name`, or (None, None) when no
    environment variable is set for it. An empty / whitespace-only value counts as
    unset, so a `export OD_FOO=` left in a sourced script means "use the default"
    rather than "set it to the empty string"."""
    candidates = [_ENV_PREFIX + name]
    if name in _ENV_LEGACY:
        candidates.append(_ENV_LEGACY[name])
    for var in candidates:
        raw = os.environ.get(var)
        if raw is not None and raw.strip() != '':
            return var, raw.strip()
    return None, None


def _env_get(name, default, parse):
    """Core override: return parse(OD_<name>) if set, else `default`.

    A value that fails to parse is a HARD error rather than a silent fallback: on
    a field laptop a typo'd export that is quietly ignored means the whole flight
    runs on a parameter nobody chose, and the log looks completely normal."""
    var, raw = _env_lookup(name)
    if raw is None:
        return default
    try:
        value = parse(raw)
    except Exception as exc:
        raise SystemExit(f"[config] {var}={raw!r} is not a valid value for "
                         f"{name} (default {default!r}): {exc}")
    _ENV_OVERRIDES.append((name, var, default, value))
    return value


def _env_str(name, default, choices=None):
    """String parameter, optionally restricted to `choices` (validated, so a
    misspelled mode fails at import instead of silently taking some other path)."""
    def parse(raw):
        if choices is not None and raw not in choices:
            raise ValueError(f"expected one of {sorted(choices)}")
        return raw
    return _env_get(name, default, parse)


_ENV_TRUE = ('1', 'true', 'yes', 'on', 'y', 't')
_ENV_FALSE = ('0', 'false', 'no', 'off', 'n', 'f')


def _env_bool(name, default):
    """Boolean parameter. Accepts 1/0, true/false, yes/no, on/off (any case)."""
    def parse(raw):
        low = raw.lower()
        if low in _ENV_TRUE:
            return True
        if low in _ENV_FALSE:
            return False
        raise ValueError("expected one of 1/0, true/false, yes/no, on/off")
    return _env_get(name, default, parse)


def _env_int(name, default, choices=None):
    """Integer parameter, optionally restricted to `choices`."""
    def parse(raw):
        value = int(raw)
        if choices is not None and value not in choices:
            raise ValueError(f"expected one of {sorted(choices)}")
        return value
    return _env_get(name, default, parse)


def _env_float(name, default):
    """Float parameter. Accepts anything float() takes, so '5', '5.0' and '5e-1'
    all work - handy when a period is exported as a plain number."""
    return _env_get(name, default, float)


def _env_device(name, default):
    """v4l2 camera device: either an OpenCV device index (0, 1, ...) or a device
    path ('/dev/video0'). A bare integer becomes an int; anything else is kept as
    the string the driver expects."""
    def parse(raw):
        return int(raw) if raw.lstrip('+-').isdigit() else raw
    return _env_get(name, default, parse)


def _env_was_set(name):
    """True when `name` was actually overridden from the environment. Used by the
    octomap rate derivation to warn about overrides it is about to discard."""
    return any(entry[0] == name for entry in _ENV_OVERRIDES)


def _env_report():
    """Print one line per applied override. Called once, after the last parameter
    below, so a field-test log records exactly which values this run did NOT take
    from the defaults in this file.

    Reports the parameter's FINAL value, not the one that was parsed out of the
    environment: the octomap rate derivation rewrites three of these after the
    fact, and a log claiming an override took effect when it did not would be
    worse than no log at all."""
    if not _ENV_OVERRIDES:
        print(f"[config] no {_ENV_PREFIX}* environment overrides - all defaults")
        return
    print(f"[config] {len(_ENV_OVERRIDES)} environment override(s):")
    for param, var, default, value in _ENV_OVERRIDES:
        effective = globals().get(param, value)
        note = '' if effective == value else f"  [DISCARDED - value in use: {effective!r}]"
        print(f"[config]   {param}: {default!r} -> {value!r}  (via {var}){note}")


# MODE selects the data source: 'airsim' (dev PC - talks to the AirSim API
# directly; the development scenario) or 'ros' (Jetson - consumes the camera/pose
# ROS topics published by publish_airsim.py in the debug scenario, or by
# publish_stereo_camera.py and a real flight controller in the deployment
# scenario). See README.md section 2.3 for all three scenarios. Override
# per-machine with the OD_MODE env var (or the legacy OBSTACLE_DETECTION_MODE)
# so both hosts can share this single config file.
#   Jetson : (default) MODE='ros'
#   Dev PC : export OD_MODE=airsim
MODE = _env_str('MODE', 'ros', choices=('airsim', 'ros'))


# Absolute path of this project folder. Used across scripts (mainly
# node_obstacle_detection.py) to locate model weights and assets relative to the repo.
ROOT_PATH = os.getenv("SWARMER_HOME", os.path.dirname(os.path.realpath(__file__)))


# Processing resolution - every perception module processes the camera feed at
# this size regardless of the source resolution. Frames entering
# SHARED_VISION_STATE are resized to (IMG_W_PROCESS, IMG_H_PROCESS), so a 4K or
# 144p drone camera and AirSim are all normalised to this before any worker runs.
IMG_W_PROCESS = _env_int('IMG_W_PROCESS', 448)
IMG_H_PROCESS = _env_int('IMG_H_PROCESS', 256)
# Camera horizontal field of view (degrees). Used by node_obstacle_detection.py to build
# the pinhole intrinsics for stereo depth and the OctoMap back-projection.
# 83 matches the Waveshare/Seeed IMX219-83 datasheet's headline FOV spec (the
# datasheet actually lists 83/73/50 degree as diagonal/horizontal/vertical - 83
# is used here directly per the module's own name/spec sheet emphasis, not the
# 73 degree horizontal figure). AirSim's camera FOV_Degrees must match this -
# see utils_airsim.JSON_TEMPLATE and settings_dataset.json.
# Together with STEREO_CAMERA_BASELINE_CMS this is half of the rig's intrinsics,
# so it is worth resolving the 83-vs-73 question against a known target on site.
FOV_D = _env_float('FOV_D', 83)

# ===== Master worker enable flags =====
# Read by airsim_demo.py and the per-task clients to choose which perception workers /
# visualizations start. (An OCTOMAP_SOURCE needing mono/stereo can auto-start those
# depth workers in node_obstacle_detection.py even if the matching DO_* flag is False.)
# The flags for the workers that are permanently off live in the Constants section
# at the bottom of this file.
DO_MONO_DEPTH_ESTIMATION = False
DO_STEREO_DEPTH_COMPUTATION = True
DO_OBS_DET = True


# ===== Debug =====
# MUST DISABLE FOR HEADLESS OPERATION (e.g. Jetson) - cv2.imshow() will crash without a display.
# Each DEBUG_*_SHOW_INPUT_TOPICS makes the matching client (node_obstacle_detection /
# node_feature_extraction / node_object_detection / node_optical_flow) cv2.imshow its incoming
# camera topic for inspection.
#
# NONE of the switches in this section is environment-overridable, on purpose:
# they open cv2 windows or write files, and an OD_* export left over in a shell
# from a previous debugging session would then crash or fill the disk on a
# headless field run. Flip them here, deliberately, and put the file back.
DEBUG_OBSTACLE_DETECTION_CLIENT_SHOW_INPUT_TOPICS = False
# One-shot sanity check of the first few camera frames actually received in ROS
# mode (not empty, right shape, not a blank/flat frame) - collects stats per
# side (left/right) for _DEBUG_CHECK_FRAME_COUNT frames, prints a short report,
# then stops checking that side. Console-only, no window - unlike
# DEBUG_OBSTACLE_DETECTION_CLIENT_SHOW_INPUT_TOPICS this works headless, so it's
# the quick answer to "is publish_stereo_camera.py (or whatever's upstream)
# actually delivering usable frames?" on a Jetson with no display attached.
DEBUG_OD_CHECK_INPUT_FRAMES = False
# Write the first _DEBUG_WRITE_VIDEO_SECONDS (20s) of received left+right
# frames, concatenated side-by-side, to one MJPG/AVI file in ROOT_PATH, for
# offline review of exactly what the camera pipeline delivered. Right is
# padded with a black frame when unavailable (mono/left-only mode), so the
# output size never changes mid-recording. MJPG/AVI is OpenCV's
# dependency-free built-in encoder - this project's OpenCV is built
# WITH_FFMPEG=OFF (see Dockerfile Step 2), so an MP4/H264 writer would likely
# fail to open here.
DEBUG_OD_WRITE_VIDEO = False
# Shows the mono depth estimation colormap in its own cv2 window (node_obstacle_detection.py).
# Also forces the mono depth estimation worker to load and run even if neither
# DO_MONO_DEPTH_ESTIMATION nor OCTOMAP_SOURCE (1/3) would otherwise start it.
DEBUG_OBSTACLE_DETECTION_CLIENT_SHOW_MONO_ESTIMATION = False
# Shows the stereo depth computation colormap in its own cv2 window (node_obstacle_detection.py).
# Also forces the stereo depth computation worker to run even if neither
# DO_STEREO_DEPTH_COMPUTATION nor OCTOMAP_SOURCE (2/3) would otherwise start it.
DEBUG_OBSTACLE_DETECTION_CLIENT_SHOW_STEREO_COMPUTATION = False
# Print the global octomap size (MB) and node count every N seconds.
# NOTE the guard in node_obstacle_detection.py is `>= 0`, so 0.0 would print on
# EVERY worker iteration - disabling this needs a negative value, not zero.
DEBUG_PRINT_OCTO_SIZE_SECS = -1.0
# Serialize the octomap to a temp_octo_<timestamp>.bt file in ROOT_PATH every N
# seconds (octovis-loadable); <= 0 = disabled.
DEBUG_SAVE_OCTOMAP = 0.0

# Print the drone position from the PX4 uORB state every N seconds; <0 = disabled.
# Console-only (no window, no file), so unlike the switches above this one is
# environment-overridable - it is the cheap way to confirm in the field that pose
# is actually arriving.
DEBUG_PRINT_DRONE_POSITION_SECS = _env_float('DEBUG_PRINT_DRONE_POSITION_SECS', -1.0)


# ===== OctoMap parameters =====
# Used by node_obstacle_detection.py's OctoMap workers (drone-centered + global) to build
# the occupancy map, and by the visualizers (vis_octomap_native.py via airsim_demo.py and
# vis_octomap_rviz.py). *_COMPUTE start the workers; the rest control resolution,
# range and point sampling. The rendering-only *_VIS_* knobs are in the Constants
# section at the bottom.

OCTOMAP_DRONE_COMPUTE = False
OCTOMAP_GLOBAL_COMPUTE = True
OCTOMAP_SOURCE = _env_int('OCTOMAP_SOURCE', 2, choices=(1, 2, 3))
                                        # 1=mono, 2=stereo, 3=stereo fallback to mono
# Far cutoff of the octomap back-projection, as a fraction of MAX_DEPTH_METERS
# (both octomap workers pass MAX_DEPTH_METERS * this as get_points_from_depth's
# max_depth_m). Replaces the old absolute OCTOMAP_RADIUS_METERS, which was the
# same per-point depth cutoff on the same depth map - deriving it keeps the one
# "how far do we trust this camera" number in MAX_DEPTH_METERS.
# The map should ingest a SHORTER range than the depth map is merely clipped to:
# depth error grows as z^2/(focal*baseline), and with the OCTOMAP_GLOBAL_TRIM_*
# limits below disabled a bad far voxel is never evicted, so far-field noise
# accumulates for the whole flight. 0.6 * 25 m = 15 m. 1.0 = map out to the full
# MAX_DEPTH_METERS; lower it if the octomap picks up noisy far-field voxels.
OCTOMAP_MAX_DEPTH_FRACTION = _env_float('OCTOMAP_MAX_DEPTH_FRACTION', 0.6)
OCTOMAP_RESOLUTION = _env_float('OCTOMAP_RESOLUTION', 3.0)   # voxel edge length, metres - default 3m
OCTOMAP_POINT_SUBSAMPLE = _env_int('OCTOMAP_POINT_SUBSAMPLE', 4)  # use every 4th pixel
# Call octree.updateInnerOccupancy() only every Nth insert instead of every one,
# and - since serializing the tree under the GIL steals time from the ROS spin
# thread - share the map on the same cadence (see the octomap workers in
# node_obstacle_detection.py). Higher = cheaper but staler inner nodes / slower
# map sharing. 1 = refresh on every insert.
# OVERRIDDEN when TOPIC_PUBLISHER_TIMER_OCTO > 0 - see "Topic publish / polling
# rates" below, which derives this from the target octomap publish period (and
# warns if it is discarding an OD_OCTOMAP_UPDATE_INNER_EVERY_N you exported).
OCTOMAP_UPDATE_INNER_EVERY_N = _env_int('OCTOMAP_UPDATE_INNER_EVERY_N', 3)

OCTOMAP_MIN_DEPTH_METERS = _env_float('OCTOMAP_MIN_DEPTH_METERS', 0.2)
# Trim the global octomap. Voxels are erased oldest-first whenever either limit is exceeded set a value <= 0 to disable
OCTOMAP_GLOBAL_TRIM_SIZE_MBS = 0.0      # cap on occupied-voxel memory (MB); above it the oldest voxels are dropped
OCTOMAP_GLOBAL_TRIM_OLD_SECONDS = 0.0   # drop voxels not re-observed for more than this many seconds
# Vertical band gating for the GLOBAL octomap, RELATIVE to the drone's current
# altitude: a detected point is never marked occupied if it lies more than this
# many centimetres below / above the drone (e.g. 500 -> 5 m). Ignores the ground
# far below or clutter far above the flight path. <0 disables that side.
# Distances are metric for stereo depth, and for mono depth once calibrated with
# OCTOMAP_MONO_DEPTH_SCALE (see below), so the same cm bands apply to both.
OCTOMAP_GLOBAL_IGNORE_BELOW_CM = -1.0
OCTOMAP_GLOBAL_IGNORE_ABOVE_CM = -1.0
# Absolute-scale calibration for mono depth. The mono worker converts Lite-Mono's
# disparity to (relative) metric depth (disp_to_depth over MONO_MIN/MAX_DEPTH_METERS);
# monocular depth has an inherent scale ambiguity, so this factor multiplies that
# depth to land it on approximate real metres. 1.0 = uncalibrated; tune against a
# known distance (or borrow the stereo scale when running OCTOMAP_SOURCE=3).
OCTOMAP_MONO_DEPTH_SCALE = 1.0

# ===== OctoMap sharing (push topic vs on-drone server) =====
# How node_obstacle_detection.py shares the octomaps it builds with other machines.
# Only effective in ROS mode (a running rclpy node is required). See README 8.18.
#   'topic'  : push - publish the serialized map on TOPIC_OCTOMAP after every
#              octomap update (original behaviour; vis_octomap_rviz.py and the
#              RViz OctoMap display consume this).
#   'server' : pull - host a GetOctomap service (octomap_msgs/srv/GetOctomap)
#              on the drone at OCTOMAP_SERVICE; consumers request the latest
#              map on demand and nothing crosses the network while nobody asks.
#   'both'   : publish AND serve (useful while migrating or debugging).
# Override per-machine without editing this file:
#   export OD_OCTOMAP_SHARE_MODE=server
OCTOMAP_SHARE_MODE = _env_str('OCTOMAP_SHARE_MODE', 'topic',
                              choices=('topic', 'server', 'both'))
# Service name used when OCTOMAP_SHARE_MODE is 'server'/'both'. '/octomap_binary'
# matches the standard octomap_server convention, so stock clients work unchanged.
OCTOMAP_SERVICE = _env_str('OCTOMAP_SERVICE', '/octomap_binary')

# ===== AirSim demo OctoMap viewer =====
# Which viewer airsim_demo.py uses for the OctoMap workers' output.
#   'native' : (default) in-process, no-ROS cv2 viewer - vis_octomap_native.py.
#              Nothing is published; no ROS 2 install required.
#   'rviz'   : publish the octree on TOPIC_OCTOMAP via a local, publish-only
#              rclpy node (node_obstacle_detection.start_octomap_rviz_bridge;
#              MODE stays 'airsim' - drone state still comes from the AirSim
#              API, only the octomap is put on ROS) and launch rviz2 with the
#              same display vis_octomap_rviz.py uses. Requires ROS 2 (rclpy,
#              octomap_msgs, rviz2) to be sourced; airsim_demo.py falls back
#              to 'native' with a logged reason if it isn't.
# Override without editing this file: export OD_AIRSIM_DEMO_VIZ=rviz
AIRSIM_DEMO_VIZ = _env_str('AIRSIM_DEMO_VIZ', 'native', choices=('native', 'rviz'))

# ===== Obstacle detection =====
# STEREO_* are read by node_obstacle_detection.py's stereo_depth_computation_worker (block
# matching + post-processing); STEREO_DEPTH_QUALITY_* drive the stereo->mono fallback
# gate. Many are adjustable live from the airsim_demo.py stereo-controls window.
# The second-order matcher / post-filter knobs are in the Constants section at the
# bottom of this file.
STEREO_NUM_DISPARITIES = 16
STEREO_USE_BASIC_BLOCK_MATCHER = True
# Stereo baseline: the physical distance (centimetres) between the two cameras'
# optical centres. This is THE single knob every stereo depth computation uses to
# scale disparity into metric depth (depth = focal * (baseline_cms / 100) /
# disparity), in both AirSim and ROS/Jetson modes - see node_obstacle_detection.py's
# stereo_depth_computation_worker and airsim_demo.py's live "Baseline (cm)" slider.
# 50 cm matches the AirSim settings template (utils_airsim.JSON_TEMPLATE: Camera1
# Y=-0.25, Camera2 Y=+0.25 -> 50 cm apart); the value below is the real onboard
# rig. Set it to the REAL MEASURED baseline of whatever rig is flying - every
# depth and every voxel scales linearly with it.
STEREO_CAMERA_BASELINE_CMS = _env_float('STEREO_CAMERA_BASELINE_CMS', 6.0)
                                  # Waveshare/Seeed IMX219-83 datasheet: Baseline Length 60mm
STEREO_SKIP_RECTIFICATION = False
STEREO_ENHANCE_WLS = True

# ===== Stereo depth de-noising / temporal smoothing =====
# Extra optional post-processing in stereo_depth_computation_worker to reduce the
# flicker and speckle in the computed depth map. Each stage has a value that fully
# disables it and restores the previous raw behaviour; set all three to their
# disable values to reproduce the exact previous behaviour.
# Pipeline order: disparity -> WLS -> (C) speckle filter -> mark disp<=0 invalid ->
# metric depth -> morph-close -> (A) temporal EMA -> quality estimate -> (B) NaN-mask.
# NOTE: these are post-filters. The single biggest noise source is usually the
# matcher itself - the default StereoBM with STEREO_NUM_DISPARITIES=16 is coarse;
# raising STEREO_NUM_DISPARITIES (multiple of 16) or switching to SGBM
# (STEREO_USE_BASIC_BLOCK_MATCHER=False) helps more than any filter here.
# (B) is STEREO_MASK_INVALID and (C) is STEREO_SPECKLE_FILTER_WINDOW /
# STEREO_SPECKLE_MAX_DIFF - both in the Constants section at the bottom.
#
# (A) Temporal exponential-moving-average smoothing across frames, computed only
# where a pixel is valid in both the current and previous frame:
#   depth = alpha*current + (1-alpha)*previous
# alpha in (0,1): lower = smoother but laggier. 1.0 disables smoothing entirely.
# Tuning: if smoothing feels laggy on the moving drone, raise alpha toward 0.6-0.7.
STEREO_TEMPORAL_SMOOTHING_ALPHA = 0.4
# Per-pixel motion guard for the EMA: if a pixel's depth changes by more than this
# many metres between frames, keep the fresh value instead of blending (so real
# motion / new obstacles are not smeared). <=0 disables the guard (always blend).
# Tuning: if edges smear on motion, lower this.
STEREO_TEMPORAL_RESET_METERS = 2.0
# Master toggle for the stereo depth quality estimate (estimate_stereo_depth_quality
# in node_obstacle_detection.py) that the STEREO_DEPTH_QUALITY_MIN_* thresholds below
# feed into. False skips the computation entirely (quality_score/valid_ratio
# stay at 0.0, quality_reason "disabled") - only meaningful to disable if nothing
# reads it, since OCTOMAP_SOURCE=3's stereo->mono fallback depends on it.
STEREO_DEPTH_QUALITY_ENABLED = True
STEREO_DEPTH_QUALITY_MIN_SCORE = 0.08
STEREO_DEPTH_QUALITY_MIN_VALID_RATIO = 0.08
# Lite-Mono pretrained variant used for monocular depth estimation (loaded by
# node_obstacle_detection.py when DO_MONO_DEPTH_ESTIMATION is True or the octomap
# source needs mono - OCTOMAP_SOURCE 1 or 3).
#   'lite-mono-small'  : 2.5M params - best efficiency
#   'lite-mono-base'   : 3.1M params - sweet spot
#   'lite-mono-8m'     : 8.7M params - highest accuracy, heaviest
MONO_DEPTH_ESTIMATION_METHOD = _env_str(
    'MONO_DEPTH_ESTIMATION_METHOD', 'lite-mono-small',
    choices=('lite-mono-small', 'lite-mono-base', 'lite-mono-8m'))
# Max depth (m) node_obstacle_detection.py clips stereo depth to. This is the single
# "how far do we trust this camera" knob and it drives three things:
#   - the clip + colormap normalisation of the stereo depth map
#   - the reference for the stereo quality gate's "too far" threshold
#     (MAX_DEPTH_METERS * STEREO_DEPTH_QUALITY_MAX_DEPTH_FRACTION)
#   - the far cutoff of the octomap back-projection (get_points_from_depth's
#     max_depth_m in both octomap workers) - it used to be a separate
#     OCTOMAP_RADIUS_METERS, but that was the same z-cutoff on the same depth map.
# Physical ceiling for the current rig: depth = focal * baseline / disparity with
# focal = IMG_W_PROCESS / (2*tan(FOV_D/2)) ~ 253 px and a 6 cm baseline gives
# ~15.2 m at 1 px of disparity, and STEREO_DEPTH_QUALITY_MIN_DISPARITY = 0.5
# already discards anything past ~30 m. Depth error grows as z^2/(focal*baseline),
# so points near this limit are coarse (~10 m of error at 25 m); lower it if the
# octomap picks up noisy far-field voxels.
MAX_DEPTH_METERS = _env_float('MAX_DEPTH_METERS', 25.0)

# ===== Frame skipping =====
# Per-worker throttle across the perception scripts (node_obstacle_detection /
# node_feature_extraction / node_object_detection / node_optical_flow): each worker processes only
# every (SKIP_FRAMES_* + 1)th frame to cap CPU/GPU load. 0 = process every frame.
# The skips belonging to permanently-off workers are in the Constants section.
SKIP_FRAMES_MONO_DEPTH_ESTIMATION = 1
SKIP_FRAMES_STEREO_DEPTH_COMPUTATION = _env_int('SKIP_FRAMES_STEREO_DEPTH_COMPUTATION', 1)
SKIP_FRAMES_OCTOMAP_DRONE = 2
# OVERRIDDEN when TOPIC_PUBLISHER_TIMER_OCTO > 0 (see "Topic publish / polling
# rates" below). NOTE the octomap workers count their own loop iterations here,
# not incoming camera frames.
SKIP_FRAMES_OCTOMAP_GLOBAL = _env_int('SKIP_FRAMES_OCTOMAP_GLOBAL', 3)

# ===== Worker delays =====
# Per-worker time.sleep (seconds) between loop iterations across the perception
# scripts - a coarse rate limiter so the worker threads yield the CPU/GPU.
# The delays belonging to permanently-off workers are in the Constants section.
WORKER_DELAY_MONO_DEPTH_ESTIMATION = 0.05
WORKER_DELAY_STEREO_DEPTH_COMPUTATION = _env_float('WORKER_DELAY_STEREO_DEPTH_COMPUTATION', 0.05)
WORKER_DELAY_OCTOMAP_DRONE = 0.05
# OVERRIDDEN when TOPIC_PUBLISHER_TIMER_OCTO > 0 - see the section right below.
WORKER_DELAY_OCTOMAP_GLOBAL = _env_float('WORKER_DELAY_OCTOMAP_GLOBAL', 0.05)


# =========================================================================
# ===== Topic publish / polling rates =====
# =========================================================================
# Every "how often does X go out / get sampled" period in one place. Kept here,
# after the SKIP_FRAMES_* and WORKER_DELAY_* blocks above, because the octomap
# target rate at the bottom of this section is DERIVED from - and overrides -
# some of them.
#
# Rates that deliberately live elsewhere, next to the hardware they belong to:
#   DIRECT_CAMERA_FPS / DIRECT_CAMERA_POLL_DELAY  - direct camera section below
#   MEASURE_CAMERA_LATENCY_PRINT_SECS             - direct camera section below
#   DEBUG_PRINT_*_SECS / DEBUG_SAVE_OCTOMAP       - debug section above
#   SKIP_FRAMES_* / WORKER_DELAY_*                - the two blocks above

# --- publish_airsim.py rclpy timer periods (seconds) ---
TOPIC_PUBLISHER_TIMER_CAMERA = _env_float('TOPIC_PUBLISHER_TIMER_CAMERA', 0.040)     # ~25 Hz - camera frames
TOPIC_PUBLISHER_TIMER_POSITION = _env_float('TOPIC_PUBLISHER_TIMER_POSITION', 0.040) # ~25 Hz - PX4 uORB state

# --- Octomap publish rate (target, seconds between shares) ---
# NOTE: unlike the three above, this is NOT an rclpy timer. The octomap is not
# published on a clock - it goes out from inside the octomap worker loop, on the
# same beat as the inner-occupancy refresh (see the octomap workers' `sync_inner`
# in node_obstacle_detection.py). So this is a TARGET that is converted into the
# three worker knobs that actually set that beat:
#
#   insert period  = (SKIP_FRAMES_OCTOMAP_GLOBAL + 1) * WORKER_DELAY_OCTOMAP_GLOBAL
#   publish period = OCTOMAP_UPDATE_INNER_EVERY_N * insert period
#
# When > 0, the block below RECOMPUTES all three of those from this target and
# overrides whatever was set above / in the OctoMap section - including anything
# exported as OD_SKIP_FRAMES_OCTOMAP_GLOBAL / OD_WORKER_DELAY_OCTOMAP_GLOBAL /
# OD_OCTOMAP_UPDATE_INNER_EVERY_N, which is why it warns when it finds one. Set
# it to 0 (or any negative value) to disable the derivation and hand-tune those
# three directly.
#
# The result is approximate by construction: the loop's own processing time
# (back-projection + insertPointCloud + optional trim/visualisation) is added on
# top of every sleep, so the achieved period is a little LONGER than asked - this
# is a floor on the period, not a guarantee. Ask for more than the Jetson can
# actually build and you simply get whatever it manages.
TOPIC_PUBLISHER_TIMER_OCTO = _env_float('TOPIC_PUBLISHER_TIMER_OCTO', 1.0)   # 1 Hz - octomap shares

# Insert cadence the derivation aims to hold while TOPIC_PUBLISHER_TIMER_OCTO
# varies. Publishing rarely must NOT mean mapping rarely: the octree still needs
# frequent insertPointCloud calls to stay dense and current, so a slower publish
# target is absorbed by OCTOMAP_UPDATE_INNER_EVERY_N (more inserts per share)
# rather than by slowing the worker down. Inserting faster than the depth workers
# produce new depth maps is wasted work, which is what caps this: the stereo
# worker yields a new map about every (SKIP_FRAMES_STEREO_DEPTH_COMPUTATION + 1)
# * WORKER_DELAY_STEREO_DEPTH_COMPUTATION = 0.1 s, so 5 Hz inserts sit one
# comfortable factor below it. Leading underscore = derivation internal, not a
# per-deployment knob.
_OCTOMAP_TARGET_INSERT_HZ = 5.0

if TOPIC_PUBLISHER_TIMER_OCTO > 0:
    _octo_hand_tuned = [n for n in ('SKIP_FRAMES_OCTOMAP_GLOBAL',
                                    'WORKER_DELAY_OCTOMAP_GLOBAL',
                                    'OCTOMAP_UPDATE_INNER_EVERY_N')
                        if _env_was_set(n)]
    if _octo_hand_tuned:
        print(f"[config] WARNING: {', '.join(_octo_hand_tuned)} set from the "
              f"environment but TOPIC_PUBLISHER_TIMER_OCTO={TOPIC_PUBLISHER_TIMER_OCTO} "
              f"> 0, so the derivation below overwrites it. Export "
              f"{_ENV_PREFIX}TOPIC_PUBLISHER_TIMER_OCTO=0 to hand-tune these three.")
    # Fastest insert period we are willing to run: the target cadence above, but
    # never slower than the publish period itself (inserting less often than we
    # share would make the extra shares meaningless).
    _insert_period_cap = min(1.0 / _OCTOMAP_TARGET_INSERT_HZ, float(TOPIC_PUBLISHER_TIMER_OCTO))
    # Loop sleep: small enough that the worker stays responsive to a fresh pose /
    # depth map and that the skip counter has something to count, but not a busy
    # spin. Lands on the other workers' 0.05 s at the default target.
    WORKER_DELAY_OCTOMAP_GLOBAL = max(0.005, min(0.05, _insert_period_cap / 2.0))
    # Inserts per share - the knob that absorbs most of the target period. FLOOR,
    # not round: it may only make the insert period LONGER than the cap when the
    # target is not a whole multiple of it, never shorter. The +1e-6 is not
    # cosmetic - binary floats make 1.0/0.2 == 4.999999999999999, so a bare floor
    # would silently return 4 inserts per share instead of 5.
    OCTOMAP_UPDATE_INNER_EVERY_N = max(1, int(
        float(TOPIC_PUBLISHER_TIMER_OCTO) / _insert_period_cap + 1e-6))
    # Back-solve the loop iterations per insert against that. NOTE this counts
    # worker loop iterations, not camera frames: the loop sleeps
    # WORKER_DELAY_OCTOMAP_GLOBAL whether it inserts or skips, so the delay and
    # the skip together set the insert period.
    SKIP_FRAMES_OCTOMAP_GLOBAL = max(0, int(round(
        float(TOPIC_PUBLISHER_TIMER_OCTO)
        / (OCTOMAP_UPDATE_INNER_EVERY_N * WORKER_DELAY_OCTOMAP_GLOBAL))) - 1)
    # Both knobs above are integers, so they can only land on a discrete grid of
    # periods (a 1.5 s target would quantise to 1.4 s). Re-derive the delay - the
    # one value here that does NOT have to be an integer - to soak up the
    # remainder, which makes the achieved period hit the target exactly instead
    # of within ~10%. It only ever moves a few ms off the 0.05 s it started at.
    WORKER_DELAY_OCTOMAP_GLOBAL = max(0.005, min(0.1, float(TOPIC_PUBLISHER_TIMER_OCTO) / (
        OCTOMAP_UPDATE_INNER_EVERY_N * (SKIP_FRAMES_OCTOMAP_GLOBAL + 1))))
    _insert_period = (SKIP_FRAMES_OCTOMAP_GLOBAL + 1) * WORKER_DELAY_OCTOMAP_GLOBAL
    print(f"[config] TOPIC_PUBLISHER_TIMER_OCTO={TOPIC_PUBLISHER_TIMER_OCTO}s -> "
          f"WORKER_DELAY_OCTOMAP_GLOBAL={WORKER_DELAY_OCTOMAP_GLOBAL:.3f} "
          f"SKIP_FRAMES_OCTOMAP_GLOBAL={SKIP_FRAMES_OCTOMAP_GLOBAL} "
          f"OCTOMAP_UPDATE_INNER_EVERY_N={OCTOMAP_UPDATE_INNER_EVERY_N} "
          f"(inserts ~{1.0 / _insert_period:.1f} Hz, "
          f"shares ~{1.0 / (OCTOMAP_UPDATE_INNER_EVERY_N * _insert_period):.2f} Hz)")

# --- Drone state sampling ---
# Caps how often utils_drone_state.py's PX4 uORB callbacks actually update the shared
# drone pose, regardless of the rate messages arrive at over the uXRCE-DDS bridge
# (VehicleAttitude in particular is commonly published at 50-250 Hz - much faster
# than any consumer here needs). The global octomap worker (OCTOMAP_GLOBAL_COMPUTE)
# only samples the pose once per its own loop - WORKER_DELAY_OCTOMAP_GLOBAL further
# decimated by SKIP_FRAMES_OCTOMAP_GLOBAL, ~5 Hz with the defaults above - so
# processing every high-rate attitude/position message would just be wasted
# callback/lock overhead on a Jetson for no gain in map accuracy. 20 Hz keeps the
# pose comfortably fresher than any current consumer needs while cutting that
# overhead; raise it only if a lower-latency consumer (e.g. a faster local planner)
# is added. <= 0 disables throttling (process every message).
DRONE_STATE_POLL_HZ = _env_float('DRONE_STATE_POLL_HZ', 20.0)


# ===== Direct (onboard) stereo camera feed =====
# Alternative to ROS topics for the camera feed. Use this when the obstacle
# detection stack runs ON the Jetson that physically carries the stereo camera:
# instead of receiving frames over the ROS topics published by publish_airsim.py,
# the consumer grabs frames straight from the onboard camera hardware with
# OpenCV (utils_camera_source.py). This removes the ROS network hop entirely.
#  - Default (False) : use ROS topics  (dev-PC / AirSim simulation path)
#  - True            : read the onboard stereo camera directly (Jetson + camera)
#
# Override per-machine without editing this file via the env var:
#   export OD_USE_DIRECT_CAMERA=1
USE_DIRECT_CAMERA = _env_bool('USE_DIRECT_CAMERA', False)

# Capture backend for the direct camera path:
#   'v4l2'      : OpenCV device indices / paths (USB / UVC stereo cameras)
#   'gstreamer' : custom GStreamer pipelines (required for Jetson CSI cameras -
#                 enables NVMM/hardware-accelerated capture via nvarguscamerasrc,
#                 which does the sensor's raw Bayer debayering in hardware.
#                 'v4l2' cannot be used for a CSI sensor: plain V4L2/OpenCV has
#                 no debayering step, so it only ever gets an unusable raw
#                 Bayer buffer - confirmed on this rig's IMX219, which only
#                 offers 'RG10' (10-bit raw Bayer) via `v4l2-ctl --list-formats-ext`,
#                 no YUYV/MJPG fallback). Default here matches this project's
#                 actual onboard rig (CSI, not USB) - override to 'v4l2' only if
#                 running on a USB/UVC stereo pair instead.
DIRECT_CAMERA_BACKEND = _env_str('DIRECT_CAMERA_BACKEND', 'gstreamer',
                                 choices=('v4l2', 'gstreamer'))

# v4l2 device indices (or paths like '/dev/video0') for the two cameras.
DIRECT_CAMERA_LEFT_DEVICE = _env_device('DIRECT_CAMERA_LEFT_DEVICE', 0)
DIRECT_CAMERA_RIGHT_DEVICE = _env_device('DIRECT_CAMERA_RIGHT_DEVICE', 1)

# GStreamer pipelines, used only when DIRECT_CAMERA_BACKEND == 'gstreamer'.
# Tune sensor-id / resolution for your CSI camera; output must be BGR for OpenCV.
# Overriding these from the shell means quoting a string full of '!' and '(' -
# use single quotes: export OD_DIRECT_CAMERA_LEFT_PIPELINE='nvarguscamerasrc ...'
DIRECT_CAMERA_LEFT_PIPELINE = _env_str('DIRECT_CAMERA_LEFT_PIPELINE', (
    "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=640,height=480,framerate=25/1 "
    "! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"))
DIRECT_CAMERA_RIGHT_PIPELINE = _env_str('DIRECT_CAMERA_RIGHT_PIPELINE', (
    "nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM),width=640,height=480,framerate=25/1 "
    "! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"))

# Do the downscale to the processing resolution (IMG_W_PROCESS x IMG_H_PROCESS)
# INSIDE the GStreamer pipeline, on the Jetson hardware scaler (nvvidconv / VIC),
# instead of resizing each frame on the CPU with cv2 afterwards. Only applies to
# the 'gstreamer' backend (CSI cameras); the 'v4l2' backend ignores it and still
# CPU-resizes. When True, publish_stereo_camera.py injects width/height into the
# nvvidconv output caps of the pipelines above, so frames arrive already at
# processing size and the CPU resize (_resize_to_process) becomes a no-op. This
# offloads the resize from the CPU and cuts the appsink/videoconvert data volume.
# Leave False if you need the pipeline to deliver full capture resolution.
DIRECT_CAMERA_GSTREAMER_HW_SCALE = _env_bool('DIRECT_CAMERA_GSTREAMER_HW_SCALE', False)

# ===== Direct camera capture resolution =====
# These values are passed to the camera backend (v4l2) as the *requested* capture
# resolution. They can be set freely to any size supported by your camera sensor
# and driver. The perception pipeline will later resize frames to IMG_W_PROCESS x
# IMG_H_PROCESS (448x256) for all workers, so the capture resolution primarily
# affects:
#   - bandwidth and latency (higher = more data to transfer/process)
#   - quality of stereo matching / feature extraction (higher can be better,
#     but diminishing returns after 640x480)
#   - compatibility with GStreamer pipelines (currently hardcoded to 1280x720 in
#     the pipeline strings; these values are only used for the v4l2 backend).
#
# Common choices:
#   640x480  (VGA)      - good balance, widely supported
#   1280x720 (HD)       - more detail, higher bandwidth
#   1920x1080 (Full HD) - max detail, heavy load
#
# If your camera does not support the requested size, the driver may either fail
# to open the device or automatically fall back to a default (often 640x480).
DIRECT_CAMERA_CAPTURE_WIDTH = _env_int('DIRECT_CAMERA_CAPTURE_WIDTH', 640)
DIRECT_CAMERA_CAPTURE_HEIGHT = _env_int('DIRECT_CAMERA_CAPTURE_HEIGHT', 480)
DIRECT_CAMERA_FPS = _env_int('DIRECT_CAMERA_FPS', 25)

# Some stereo cameras expose a single device that delivers a side-by-side
# (left|right) frame. Set True to open only DIRECT_CAMERA_LEFT_DEVICE and split
# each frame down the middle into the left/right images.
DIRECT_CAMERA_SIDE_BY_SIDE = False
# Idle delay (s) of the capture loop between polls.
DIRECT_CAMERA_POLL_DELAY = _env_float('DIRECT_CAMERA_POLL_DELAY', 0.002)

# ===== Camera feed latency measurement (node_obstacle_detection.py) =====
# When MEASURE_CAMERA_LATENCY_PRINT_SECS > 0, node_obstacle_detection.py measures and
# periodically prints the camera # feed acquisition latency for
# whichever source is active, so the two transport paths can be compared:
#   - ROS topics    (USE_DIRECT_CAMERA = False): header.stamp -> receive time
#                    (requires the dev PC and Jetson clocks to be NTP-synced for
#                     the absolute number to be meaningful)
#   - Direct camera (USE_DIRECT_CAMERA = True) : cv2 grab -> frame-ready time
MEASURE_CAMERA_LATENCY_PRINT_SECS = _env_float('MEASURE_CAMERA_LATENCY_PRINT_SECS', 5.0)

# ===== ROS2 topics =====
# ROS 2 topic names shared by the publisher (publish_airsim.py) and the Jetson clients
# (node_obstacle_detection / node_feature_extraction / node_object_detection / node_optical_flow);
# test_conn.py also validates these exist. USE_COMPRESSED_IMAGE_TOPICS picks raw vs
# JPEG image topics. Topic NAMES live here; the matching publish PERIODS
# (TOPIC_PUBLISHER_TIMER_*) are grouped in "Topic publish / polling rates" above.
# A renamed topic must be exported on BOTH machines - the publisher and every
# consumer read the same parameter out of this one file.
TOPIC_CAMERA_LEFT = _env_str('TOPIC_CAMERA_LEFT', '/camera/left/image_raw')
TOPIC_CAMERA_RIGHT = _env_str('TOPIC_CAMERA_RIGHT', '/camera/right/image_raw')
USE_COMPRESSED_IMAGE_TOPICS = _env_bool('USE_COMPRESSED_IMAGE_TOPICS', True)
TOPIC_CAMERA_LEFT_COMPRESSED = _env_str('TOPIC_CAMERA_LEFT_COMPRESSED', '/camera/left/image_compressed')
TOPIC_CAMERA_RIGHT_COMPRESSED = _env_str('TOPIC_CAMERA_RIGHT_COMPRESSED', '/camera/right/image_compressed')
TOPIC_OCTOMAP = _env_str('TOPIC_OCTOMAP', '/octo_full')
# NOTE: there is deliberately no TOPIC_DRONE_POSITION here any more. The drone
# pose travels ONLY on the PX4 uORB topics below (/fmu/out/*), in simulation as
# well as on the real drone - publish_airsim.py translates AirSim's state
# straight into them. '/drone/position' was this project's own
# geometry_msgs/PoseStamped stand-in for a flight controller; the translation is
# now in-process (utils_airsim.airsim_state_to_px4), so it never reaches the ROS
# network and is not a configurable topic.

# ===== PX4 uORB state estimation =====
# node_obstacle_detection.py ALWAYS sources the drone position & orientation from
# PX4's uORB messages exposed over the uXRCE-DDS bridge (px4_msgs). This is not a
# mode any more - there is deliberately no toggle:
#   - on the real drone the messages come from PX4's EKF2 over uXRCE-DDS;
#   - in simulation publish_airsim.py synthesises the same four messages from
#     AirSim's state (see the "AirSim -> PX4 uORB state translation" section of
#     utils_airsim.py), so the topic names AND message types are identical.
# One topic set, one message type per quantity, in simulation and in the field -
# which is what the SWARMER integration expects to see in a rosbag.
#
# Requires the px4_msgs package to be built into the workspace, on BOTH machines
# (publish_airsim.py needs it to publish, node_obstacle_detection.py to
# subscribe). There is no fallback: without px4_msgs there is no pose source at
# all, so both sides fail loudly at startup rather than run on a pose that never
# leaves the origin and quietly produce a garbage octomap.
#
# uORB VehicleLocalPosition - NED position (m)
TOPIC_PX4_VEHICLE_LOCAL_POSITION = _env_str('TOPIC_PX4_VEHICLE_LOCAL_POSITION',
                                            '/fmu/out/vehicle_local_position')
# uORB VehicleAttitude - body(FRD)->NED quaternion [w,x,y,z]
TOPIC_PX4_VEHICLE_ATTITUDE = _env_str('TOPIC_PX4_VEHICLE_ATTITUDE',
                                      '/fmu/out/vehicle_attitude')
# uORB VehicleGlobalPosition - lat/lon/alt
TOPIC_PX4_VEHICLE_GLOBAL_POSITION = _env_str('TOPIC_PX4_VEHICLE_GLOBAL_POSITION',
                                             '/fmu/out/vehicle_global_position')
# uORB EstimatorStatus - EKF health/accuracy
TOPIC_PX4_ESTIMATOR_STATUS = _env_str('TOPIC_PX4_ESTIMATOR_STATUS',
                                      '/fmu/out/estimator_status')
# (How often these callbacks are allowed to update the shared pose is
# DRONE_STATE_POLL_HZ, in "Topic publish / polling rates" above.)


# Last environment-overridable parameter is above: report what this run changed.
_env_report()


# =========================================================================
# ===== Constants (design choices - not field-tunable) =====
# =========================================================================
# Everything below is deliberately NOT exposed as an OD_* environment variable.
# Each entry is here because it is either a physical/algorithmic invariant, an
# internal of a single function, a dev-PC-only rendering knob, or a parameter of
# a worker that is switched off. Promote one back up into the sections above (and
# into tools/env_vars.sh) if it turns out to need adjusting in the field.

# --- Master flags for the permanently-off workers ---
# All consumed by airsim_demo.py (and the matching node_*.py, which never starts
# with these False), i.e. the dev-PC demo path only.
DO_RADARS = False
DO_FEAT_DET = False
DO_FEAT_DESC = False
DO_FEAT_ALIGN = False
DO_OBJ_DET = False
DO_OPTICAL_FLOW = False
DO_DEPTH_SENSOR_WORKER = False

# --- Debug switches for those workers / for the test scripts ---
DEBUG_FEATURE_EXTRACTION_CLIENT_SHOW_INPUT_TOPICS = False
DEBUG_OBJECT_DETECTION_CLIENT_SHOW_INPUT_TOPICS = False
DEBUG_OPTICAL_FLOW_CLIENT_SHOW_INPUT_TOPICS = False
# Shows each Lite-Mono variant's estimated depth colormap in a cv2 window while
# test_lite_mono.py benchmarks it.
DEBUG_TEST_MONO_SHOW_ESTIMATION = False
# Print every Nth left frame received (node_obstacle_detection.py); 0 = disabled.
DEBUG_OBSTACLE_DETECTION_CLIENT_PRINT_LEFT_FRAME_COUNT = 0

# --- OctoMap internals ---
OCTOMAP_NODE_SIZE_BYTES = 72            # approximate size of a ColorOcTreeNode -
                                        # a property of the octomap library, not
                                        # of this deployment
OCTOMAP_SKIP_IF_OCCUPIED = True         # algorithmic choice in the insert path
OCTOMAP_VOXEL_DOWNSAMPLE = False
# Depth range used to turn Lite-Mono's sigmoid disparity into metric depth
# (Monodepth2/Lite-Mono disp_to_depth convention; this is the NETWORK'S TRAINING
# RANGE, not a per-deployment knob). Absolute scale is set by
# OCTOMAP_MONO_DEPTH_SCALE in the OctoMap section above - adjust that, not these.
OCTOMAP_MONO_MIN_DEPTH_METERS = 0.1
OCTOMAP_MONO_MAX_DEPTH_METERS = 100.0

# --- OctoMap rendering (vis_octomap_native.py, dev PC only) ---
OCTOMAP_DRONE_VIS_ZOOM = 25.0
OCTOMAP_GLOBAL_VIS_ZOOM = 25.0
OCTOMAP_GLOBAL_VIS_TILT = 15

# --- Stereo matcher / post-filter second-order knobs ---
# Live-adjustable from the airsim_demo.py stereo-controls window on the dev PC,
# which is where they are meant to be tuned. The first-order matcher choices
# (STEREO_NUM_DISPARITIES, STEREO_USE_BASIC_BLOCK_MATCHER) are in the obstacle
# detection section above.
STEREO_MIN_DISPARITIES = 0
STEREO_SGBM_BLOCK_SIZE = 11
STEREO_SGBM_UNIQUENESS = 1
STEREO_SGBM_SPECKLE_WINDOW = 100
STEREO_SGBM_SPECKLE_RANGE = 2
STEREO_MORPH_CLOSE_KERNEL = 3
# (B) Mask unmatched pixels. The stereo matcher returns <=0 disparity where it
# found no correspondence; previously those pixels were forced to a tiny disparity
# and so read as MAX_DEPTH (a flashing far wall). When True they are marked as NaN
# (no data) instead - excluded from the octomap back-projection and rendered dark
# in the colormap - which is both correct and much less flickery. False = old
# behaviour (invalid pixels clamped to MAX_DEPTH_METERS).
STEREO_MASK_INVALID = True
# (C) Speckle rejection (cv2.filterSpeckles) on the raw disparity: connected blobs
# smaller than this many pixels whose disparity differs by more than
# STEREO_SPECKLE_MAX_DIFF (disparity units) are invalidated. 0 disables it.
STEREO_SPECKLE_FILTER_WINDOW = 50
STEREO_SPECKLE_MAX_DIFF = 2.0

# --- Stereo quality score internals ---
# Shape parameters of estimate_stereo_depth_quality() itself (one function, one
# file). The two thresholds that actually decide the OCTOMAP_SOURCE=3 stereo->mono
# fallback - STEREO_DEPTH_QUALITY_MIN_SCORE / _MIN_VALID_RATIO - are above.
STEREO_DEPTH_QUALITY_MIN_IMAGE_DIFF = 1.0
STEREO_DEPTH_QUALITY_MIN_TEXTURE_STD = 3.0
STEREO_DEPTH_QUALITY_MIN_DISPARITY = 0.5
STEREO_DEPTH_QUALITY_MAX_DEPTH_FRACTION = 0.8
STEREO_DEPTH_QUALITY_SAMPLE_STEP = 8

# --- Optical flow (DO_OPTICAL_FLOW is False) ---
# Used by node_optical_flow.py's OpticalFlow worker: flow algorithm selection plus
# magnitude clipping / temporal smoothing of the computed flow field.
OPTICAL_FLOW_TEMPORAL_SMOOTHING_ENABLED = True
OPTICAL_FLOW_MAX_DISPLACEMENT = 64.0
OPTICAL_FLOW_METHOD = 'dis_m'
SKIP_FRAMES_OPTICAL_FLOW = 1
WORKER_DELAY_OPTICAL_FLOW = 0.05

# --- Feature extraction (DO_FEAT_* are False) ---
# Used by node_feature_extraction.py: keypoint detector / descriptor algorithm and
# the max number of matches kept during frame-to-frame feature alignment.
FEAT_DET_METHOD = 'ORB'
FEAT_DESC_METHOD = 'ORB'
FEAT_ALIGN_K = 100
SKIP_FRAMES_FEAT_EXTRACTION = 1
WORKER_DELAY_FEAT_DET = 0.05
WORKER_DELAY_FEAT_DESC = 0.05
WORKER_DELAY_FEAT_ALIGN = 0.05

# --- Object detection (DO_OBJ_DET is False) ---
# Used by node_object_detection.py: the Ultralytics YOLO weights file loaded for inference.
OBJECT_DETECTION_YOLO_MODEL = 'yolo26n.pt'
SKIP_FRAMES_OBJ_DET = 1
WORKER_DELAY_OBJ_DET = 0.05

# --- Depth sensor worker (DO_DEPTH_SENSOR_WORKER is False) ---
WORKER_DELAY_DEPTH_SENSOR = 0.05


# ===== Shared worker state objects =====
# Thread-safe containers each perception worker writes its latest result into, and
# airsim_demo.py / the clients read for visualization & runtime metrics. Instantiated in
# the respective scripts (node_obstacle_detection.py, node_feature_extraction.py, etc.).
class ObjDetState:
    def __init__(self):
        self.boxes = []
        self.process_time_ms = 0.0
        self.lock = threading.Lock()
        self.running = True

class FeatDetState:
    def __init__(self):
        self.keypoints = []
        self.prev_keypoints = None
        self.kp_version = 0
        self.process_time_ms = 0.0
        self.lock = threading.Lock()
        self.running = True

class FeatDescState:
    def __init__(self):
        self.descriptors = None
        self.prev_descriptors = None
        self.desc_version = 0
        self.process_time_ms = 0.0
        self.lock = threading.Lock()
        self.running = True

class FeatAlignState:
    def __init__(self):
        self.aligned_kps = []
        self.aligned_descs = []
        self.process_time_ms = 0.0
        self.lock = threading.Lock()
        self.running = True

class DepthSensorState:
    def __init__(self):
        self.depth_colormap = None
        self.process_time_ms = 0.0
        self.lock = threading.Lock()
        self.running = True

class MonoDepthEstState:
    def __init__(self):
        self.depth_colormap = None
        self.depth_map = None
        self.process_time_ms = 0.0
        self.lock = threading.Lock()
        self.running = True

class StereoDepthCompState:
    def __init__(self):
        self.depth_colormap = None
        self.depth_map = None
        self.quality_score = 0.0
        self.quality_valid_ratio = 0.0
        self.quality_reason = "not computed"
        self.process_time_ms = 0.0
        self.lock = threading.Lock()
        self.running = True

class OpticalFlowState:
    def __init__(self):
        self.flow = None
        self.prev_gray = None
        self.process_time_ms = 0.0
        self.lock = threading.Lock()
        self.running = True

class VisionFrameState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame1 = None
        self.frame2 = None
        self.depth_uint8 = None
        self.cam1_info = None
        self.cam2_info = None
        self.frame_id = 0

    def _to_process_size(self, frame):
        """Resize an incoming frame to the configured processing resolution so
        every worker sees (IMG_W_PROCESS, IMG_H_PROCESS) regardless of whether the
        source is AirSim, the onboard camera, or a ROS topic at any resolution."""
        if frame is None:
            return None
        if frame.shape[1] == IMG_W_PROCESS and frame.shape[0] == IMG_H_PROCESS:
            return frame
        return cv2.resize(frame, (IMG_W_PROCESS, IMG_H_PROCESS))

    def set_frame1(self, frame):
        """Thread-safe single-frame setter (resized to processing size). Used by
        the ROS image callbacks, which receive left/right frames independently."""
        with self.lock:
            self.frame1 = self._to_process_size(frame)

    def set_frame2(self, frame):
        with self.lock:
            self.frame2 = self._to_process_size(frame)

    def update_frames(self, frame1, frame2):
        self.frame1 = self._to_process_size(frame1)
        self.frame2 = self._to_process_size(frame2)
        self.frame_id += 1


# Single shared camera-frame buffer (at processing resolution): every camera source
# writes here and every perception worker reads from it. See VisionFrameState above.
SHARED_VISION_STATE = VisionFrameState()

class OctomapDroneState:
    def __init__(self):
        self.process_time_ms = 0.0
        self.visualization_process_time_ms = 0.0
        self.memory_mb = 0.0
        self.num_nodes = 0
        self.lock = threading.Lock()
        self.running = True

class OctomapGlobalState:
    def __init__(self):
        self.process_time_ms = 0.0
        self.visualization_process_time_ms = 0.0
        self.memory_mb = 0.0
        self.num_nodes = 0
        self.lock = threading.Lock()
        self.running = True



# AirSim launch/settings helpers moved to utils_airsim.py (imported by airsim_demo.py and
# publish_airsim.py). Kept out of here so config.py stays pure configuration.
