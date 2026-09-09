"""Collect a synthetic multi-baseline stereo-depth dataset from AirSim.

Captures, in a single combined simGetImages call on one AirSim connection
(the whole process uses exactly one connection - see fetch_image and
try_calibrate_depth_lut's docstrings for why that matters):
  - left RGB   (the first camera in --settings, Scene)         -> shared reference
  - right RGB  (every other camera in --settings, Scene)        -> one per baseline
  - left depth (DepthPerspective, non-float - see below)        -> ground truth
plus the left-camera pose and body kinematics per frame, saved as a *sequence*
so that temporal models can use motion between consecutive frames. Disk
writes (PNG + npy) happen on a background thread (save_worker) so they don't
add to the per-frame AirSim RPC fetch latency, which measurement showed as
the actual FPS-capping cost, not disk I/O or the preview window - see
save_worker's docstring for why only disk I/O, not the fetch itself, is
threaded.

Depth deserves explanation. AirSim's DepthPerspective *float* capture
(pixels_as_float=True) has been confirmed to hang its RPC indefinitely on
this build - not intermittently in the usual sense, but capped at ~1
success per AirSim session before permanently hanging on every attempt
after that, regardless of client connection strategy (fresh connections
per attempt, one persistent connection, a different render backend -
Vulkan vs OpenGL4 - none of it changed the outcome). That points at
something server-side (the AirSim/Unreal process itself), which no
client-side workaround can route around. Non-float capture
(pixels_as_float=False) has been reliable in every test, including
combined with Scene requests in one call, so it's the only depth capture
used per-frame; float capture is used exactly once per --out directory,
opportunistically, purely to *calibrate* non-float's 8-bit
depth-visualization encoding (no documented formula back to metric meters)
against real distances - see build_depth_lut/try_calibrate_depth_lut. The
resulting lookup table is cached at <out>/depth_lut.npy and reused by every
subsequent ride writing to that same --out, so only the first ride in a
campaign needs a lucky float capture at all - and if that one hangs, it
hangs the whole process (same as any other stuck fetch_image call),
recovered externally by the process-level `timeout` in
run_dataset_campaign.sh, not from inside this script. Depth quality is
bounded by 8-bit quantization (256 distinct levels across whatever range
the calibration frame covered) rather than being exact - a real accuracy
trade accepted in exchange for actually getting depth data reliably. The
raw DepthPerspective range (Euclidean R) - whether from the calibration
float sample or looked up from the LUT - is converted to planar depth
Z = R / sqrt(1 + dx^2 + dy^2) once per ride, since that's the quantity the
disp = fx*B/depth formula below actually assumes.

--depth-decode packed is an alternate, opt-in decoding of the same
non-float capture (ported from airsim_recorder.py): instead of the LUT's
R=G=B-grayscale assumption, it treats the 3 channels as a 24-bit millimeter
integer, needing no float capture at all and no 8-bit quantization - but the
channel packing isn't documented by AirSim (channel order is auto-detected
by variance, see pick_packed_order) and hasn't been cross-validated against
the LUT path on this build, hence opt-in rather than the default. See
depth_packed_to_meters.

Camera names, count, and baselines are NOT hardcoded: they are read from the
AirSim settings JSON given via --settings (default settings_dataset.json),
whose "Cameras" dict's first entry is the left/reference camera and every
other entry is a right camera at its own baseline (its Y offset from the
left camera, in meters). This lets one recording produce several baselines
at once, so downstream users pick whichever folder matches their own rig's
baseline instead of re-recording per baseline.

Output layout:
  <out>/depth_lut.npy         shared across every ride writing to this --out -
                               (256,) float32 lookup table, non-float depth
                               byte value -> calibrated meters (see above)
  <out>/seq_<timestamp>_<tag>/
      calib.json              fx, fy, cx, cy, width, height, fps, and one
                               stereo_pairs[] entry per right camera (camera
                               name, baseline_m, image dir)
      conditions.json          env, weather, wind, time-of-day, and altitude-band
                               target requested for this ride (--env, --weather, --wind,
                               --time-of-day, --altitude), for per-ride condition logging
      poses.csv               per-frame timestamp + left-camera pose + body velocities
      left/000000.png         left RGB
      depth/000000.npy        left planar depth, float32 meters (ZFar-clamped by
                               AirSim), looked up from depth_lut.npy and converted
                               from DepthPerspective range to planar - 8-bit-quantized
                               accuracy, not exact (see the note above)
      right_0038mm/000000.png right RGB for the ~3.75cm-baseline camera
      right_0063mm/000000.png right RGB for the ~6.3cm-baseline camera
      ...                     one right_<baseline_mm>mm/ dir per right camera

Disparity ground truth is derived at load time, per stereo pair:
  disp = fx * baseline / depth

Usage (always launches the AirSim binary itself via utils_airsim, killing any
previously-running instance first):
    python collect_dataset.py --out ~/datasets/airsim_stereo --tag nh --env AirSimNH

Fly with the arrow keys / PageUp / PageDown (same bindings as publish_airsim.py).
Recording stops after --duration seconds or on Ctrl+C.

A small preview window (left RGB / first right RGB / colorized depth, like
airsim_recorder.py's) is shown during the recording loop unless --no-preview
is passed; ESC in that window also stops recording early, same as Ctrl+C.
"""

import argparse
import csv
import json
import math
import os
import queue
import random
import signal
import sys
import threading
import time

import numpy as np
import cv2

try:
    import airsim
except ImportError:
    sys.exit("The 'airsim' package is required (pip install airsim --no-build-isolation). "
             "Activate the environment you use for publish_airsim.py.")

# Reuse the project's sim-launch helpers (parent directory).
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(PROJECT_ROOT)
import utils_airsim

DEFAULT_SETTINGS = os.path.join(PROJECT_ROOT, "json_templates", "settings_dataset.json")
FOV_DEGREES = 83.0  # must match settings.json CaptureSettings (real rig: IMX219-83)
PREVIEW_HEIGHT_PX = 200  # small on purpose - this is a live sanity-check, not a real viewer
PREVIEW_MAX_DEPTH_M = 30.0  # preview colormap FALLBACK only, used when a frame has no valid depth pixels to
                             # percentile off of (see the preview block below) - not the normal case
PREVIEW_GAMMA = 0.5  # depth**PREVIEW_GAMMA before colorizing - 1.0 is linear (washes out near-field
                      # contrast), smaller values push more contrast toward the near field at the cost of
                      # far-field contrast (0 would be maximal, i.e. log-like); 0.5 (sqrt) is a middle ground

# AirSim environment binaries (Section 2.1), keyed by name for --env. Every
# entry follows the same <Env>/LinuxNoEditor/<Env>.sh layout under
# utils_airsim.AIRSIM_BINARY_ROOT.
ENV_SCRIPTS = {
    name: os.path.join(utils_airsim.AIRSIM_BINARY_ROOT, name, "LinuxNoEditor", f"{name}.sh")
    for name in ("AirSimNH", "AbandonedPark", "Africa_Savannah", "Blocks",
                 "LandscapeMountains", "ZhangJiajie", "TrapCam")
}

def load_cameras(settings_path):
    """Read the Cameras dict from an AirSim settings JSON and split it into
    (left_name, [(right_name, nominal_baseline_m), ...]), preserving JSON
    order: the first camera is the left/reference camera, every other camera
    is a right camera whose distance from the left camera (from its X/Y/Z in
    the same file) is its baseline."""
    with open(settings_path) as f:
        settings = json.load(f)
    cams = settings["Vehicles"]["Drone1"]["Cameras"]
    names = list(cams.keys())
    if len(names) < 2:
        sys.exit(f"{settings_path}: need a left camera plus at least one right "
                  f"camera, found {names}")
    left_name = names[0]
    left_pos = (cams[left_name]["X"], cams[left_name]["Y"], cams[left_name]["Z"])
    rights = []
    for name in names[1:]:
        pos = (cams[name]["X"], cams[name]["Y"], cams[name]["Z"])
        rights.append((name, math.dist(left_pos, pos)))
    return left_name, rights

KEYS = {'up': False, 'down': False, 'left': False, 'right': False,
        'pageup': False, 'pagedown': False}


def _make_key_handlers():
    from pynput import keyboard

    mapping = {keyboard.Key.up: 'up', keyboard.Key.down: 'down',
               keyboard.Key.left: 'left', keyboard.Key.right: 'right',
               keyboard.Key.page_up: 'pageup', keyboard.Key.page_down: 'pagedown'}

    def on_press(key):
        if key in mapping:
            KEYS[mapping[key]] = True

    def on_release(key):
        if key in mapping:
            KEYS[mapping[key]] = False

    return keyboard.Listener(on_press=on_press, on_release=on_release)


def decode_rgb(response):
    img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    if img.size != response.height * response.width * 3:
        return None
    return img.reshape(response.height, response.width, 3)  # BGR


def decode_depth(response):
    depth = np.asarray(response.image_data_float, dtype=np.float32)
    if depth.size != response.height * response.width:
        return None
    return depth.reshape(response.height, response.width)


def decode_depth_nonfloat(response):
    """Decodes AirSim's non-float depth-visualization response: a grayscale
    image (R=G=B) with no documented mapping back to metric meters - see
    build_depth_lut for how that mapping gets derived empirically instead."""
    img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    if img.size != response.height * response.width * 3:
        return None
    return img.reshape(response.height, response.width, 3)[:, :, 0]


def decode_depth_raw_uint8(response):
    """Like decode_depth_nonfloat, but keeps all 3 channels instead of
    reducing to channel 0 - needed by the --depth-decode packed path below,
    which (unlike decode_depth_nonfloat's R=G=B assumption) reads all three
    channels as one 24-bit value."""
    img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    if img.size != response.height * response.width * 3:
        return None
    return img.reshape(response.height, response.width, 3)


def depth_packed_to_meters(img_uint8, order):
    """Alternate decoding of AirSim's non-float depth response, ported from
    airsim_recorder.py's decode_depth_uint8: instead of treating the 3
    channels as a plain R=G=B grayscale visualization needing an empirical
    LUT (build_depth_lut), this assumes AirSim packed a 24-bit millimeter
    integer across them (mm = ch0 + ch1*256 + ch2*65536). If accurate on a
    given build, it recovers metric depth directly from every per-frame
    capture - no LUT, no one-time float calibration, and far more precision
    (mm vs the LUT path's 256 quantization levels) - but the channel-to-RGB
    mapping isn't documented (airsim_recorder.py auto-detects it rather than
    assuming), and this hasn't been cross-validated against this project's
    own float-calibrated LUT. Hence --depth-decode packed is opt-in, not the
    default: compare its depth/ output against 'lut' before trusting it for
    a real campaign.
    order: 'rgb' (ch0=R, ch1=G, ch2=B) or 'bgr' (ch0=B, ch1=G, ch2=R) - see
    pick_packed_order."""
    if order == 'rgb':
        r, g, b = (img_uint8[:, :, i].astype(np.float64) for i in (0, 1, 2))
    else:  # bgr
        r, g, b = (img_uint8[:, :, i].astype(np.float64) for i in (2, 1, 0))
    return ((r + g * 256.0 + b * 65536.0) / 1000.0).astype(np.float32)


def pick_packed_order(img_uint8):
    """One-shot auto-detection of which channel order depth_packed_to_meters
    should use, ported from airsim_recorder.py's choose_best_order: decode
    both orders and keep whichever gives higher depth variance over a
    plausible 0-1000m range, on the theory that the wrong order scrambles
    the 24-bit value into near-random noise while the right order reproduces
    smooth scene depth."""
    best_order, best_var = 'bgr', -1.0
    for order in ('rgb', 'bgr'):
        depth = depth_packed_to_meters(img_uint8, order)
        valid = (depth > 0) & (depth < 1000)
        var = np.var(depth[valid]) if np.any(valid) else 0.0
        if var > best_var:
            best_order, best_var = order, var
    return best_order


def build_depth_lut(gray_uint8, meters_float):
    """Empirically maps each of the 256 possible non-float depth-byte values
    to a representative metric depth, by averaging the true float depth of
    every pixel sharing that byte value in one paired calibration capture
    (same camera, same instant, requested as both pixels_as_float=True and
    pixels_as_float=False). Byte values with no observed pixels in this one
    frame are filled in by linear interpolation between their nearest
    populated neighbors. Returns a (256,) float32 array, or None if the
    capture had too little depth variation to calibrate more than one bin."""
    counts = np.zeros(256, dtype=np.int64)
    sums = np.zeros(256, dtype=np.float64)
    flat_gray = gray_uint8.ravel().astype(np.int64)
    flat_meters = meters_float.ravel().astype(np.float64)
    np.add.at(sums, flat_gray, flat_meters)
    np.add.at(counts, flat_gray, 1)
    valid = counts > 0
    if valid.sum() < 2:
        return None
    lut = np.empty(256, dtype=np.float64)
    lut[valid] = sums[valid] / counts[valid]
    idx = np.arange(256)
    lut = np.interp(idx, idx[valid], lut[valid])
    return lut.astype(np.float32)


def try_calibrate_depth_lut(client, camera_name):
    """One-shot attempt to build a depth_lut (see build_depth_lut) by
    capturing DepthPerspective as both float and non-float, back to back, on
    the SAME shared connection everything else in this script uses - no
    separate connection, no background thread. The float half is the same
    fragile capture that hangs on this build; if it does, this call just
    hangs too (like any other fetch_image call), and recovery happens one
    level up via the process-level `timeout` in run_dataset_campaign.sh,
    not from inside this script. Returns None on any decode/shape mismatch;
    the caller is expected to fall back to a previously-saved LUT if one
    exists rather than treat None as fatal."""
    try:
        float_req = airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False)
        nonfloat_req = airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, False, False)
        meters = decode_depth(fetch_image(client, float_req))
        gray = decode_depth_nonfloat(fetch_image(client, nonfloat_req))
        if meters is not None and gray is not None and meters.shape == gray.shape:
            return build_depth_lut(gray, meters)
    except Exception:
        pass
    return None


def send_flight_command(client, speed, z_speed, turn_speed, cmd_duration):
    """Reads the live KEYS state and issues one moveByVelocityBodyFrameAsync
    command accordingly. Shared between the pre-recording countdown and the
    main recording loop so keyboard flight works identically in both."""
    vx, vz, yaw = 0.0, 0.0, 0.0
    if KEYS['up']:       vx = speed
    if KEYS['down']:     vx = -speed
    if KEYS['left']:     yaw = -turn_speed
    if KEYS['right']:    yaw = turn_speed
    if KEYS['pageup']:   vz = -z_speed
    if KEYS['pagedown']: vz = z_speed
    client.moveByVelocityBodyFrameAsync(vx, 0, vz, duration=cmd_duration,
                                        yaw_mode=airsim.YawMode(True, yaw),
                                        vehicle_name="Drone1")


def fetch_image(client, request):
    """simGetImages([request]) for one request. Used for the low-frequency
    calibration paths (the one-time stationary baseline capture, and
    try_calibrate_depth_lut's float+non-float pair, kept as two separate
    calls specifically so a hung float request can't drag down the
    reliable non-float one) - NOT for the per-frame recording loop, which
    combines everything into a single simGetImages call for speed now that
    depth capture there is non-float only.

    A hang here is fatal to the whole process by design (a threaded
    timeout+reconnect on this shared connection was tried and made things
    worse - crashes with "IOLoop is already running" instead of recovering,
    since the airsim client's underlying msgpackrpc/tornado stack isn't
    safe for a second blocking call while a prior one is stuck in a
    background thread on the same connection - see git history). Recovery
    from a hang happens one level up, via the process-level `timeout`
    wrapping this whole script in run_dataset_campaign.sh - not from inside
    this script, and not by giving depth capture its own separate
    connection/thread either, since that reintroduces the exact
    two-connections-at-once risk this avoids."""
    return client.simGetImages([request])[0]


def save_worker(save_queue, write_stats):
    """Background disk-write thread for the recording loop's per-frame PNG +
    npy writes. Deliberately the ONLY thing threaded off the main loop:
    measurement (see main's per-stage timing) showed the AirSim RPC fetch at
    ~134ms/frame vs disk save at ~37ms/frame, and unlike the fetch (see
    fetch_image's docstring on why THAT stays single-threaded - a second
    thread touching the same AirSim connection previously crashed with
    "IOLoop is already running"), disk I/O never touches the AirSim
    connection at all, so overlapping it with the next frame's fetch is
    safe. write_stats accumulates real write time (n, seconds) for the
    "Done:" summary; only this thread writes to it, so the main thread can
    read it race-free once it has confirmed (via save_queue.join()) that
    this thread has consumed every item, including the None sentinel that
    ends it."""
    while True:
        item = save_queue.get()
        if item is None:
            save_queue.task_done()
            return
        seq_dir, name, left, depth, rights, right_dirs = item
        t0 = time.monotonic()
        try:
            cv2.imwrite(os.path.join(seq_dir, "left", name + ".png"), left)
            np.save(os.path.join(seq_dir, "depth", name + ".npy"), depth)
            for right_img, dir_name in zip(rights, right_dirs):
                cv2.imwrite(os.path.join(seq_dir, dir_name, name + ".png"), right_img)
        except Exception as e:
            print(f"  [warn] background save failed for frame {name}: {e}")
        else:
            write_stats["n"] += 1
            write_stats["seconds"] += time.monotonic() - t0
        save_queue.task_done()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="dataset root directory")
    ap.add_argument("--tag", default="ride", help="sequence name suffix (environment name)")
    ap.add_argument("--fps", type=float, default=10.0, help="capture rate (Hz)")
    ap.add_argument("--duration", type=float, default=10.0, help="recording length in seconds")
    ap.add_argument("--env", choices=sorted(ENV_SCRIPTS), default=None,
                    help="which AirSim environment binary to launch "
                         "(default: whichever ENV_SCRIPT is uncommented in utils_airsim.py)")
    ap.add_argument("--min-motion", type=float, default=0.0,
                    help="skip saving a frame if the camera moved less than this many meters "
                         "since the last saved frame (0 = save everything)")
    ap.add_argument("--settings", default=DEFAULT_SETTINGS,
                    help="AirSim settings JSON describing the camera rig: first camera is "
                         "left/reference, every other camera is a right camera at its own "
                         "baseline (default: settings_dataset.json, 1 right camera)")
    ap.add_argument("--weather", choices=["clear", "rain", "fog", "dust"], default=None,
                    help="weather condition to set before recording (default: leave AirSim's "
                         "current weather as-is)")
    ap.add_argument("--weather-intensity", type=float, default=0.6,
                    help="rain/fog/dust intensity, 0-1 (default: 0.6, ignored for --weather clear)")
    ap.add_argument("--wind", type=float, default=None,
                    help="horizontal wind speed in m/s, applied in a random heading (default: "
                         "leave AirSim's current wind as-is; pass 0 to explicitly clear it)")
    ap.add_argument("--time-of-day", default=None,
                    help="AirSim time-of-day as 'YYYY-MM-DD HH:MM:SS' (default: leave as-is)")
    ap.add_argument("--altitude", type=float, default=None,
                    help="altitude band in meters AGL to climb/descend to right after takeoff, "
                         "e.g. 3 (under-canopy), 12 (cruise), 25 (long-range); default: whatever "
                         "altitude takeoff leaves the drone at.")
    ap.add_argument("--speed", type=float, default=None,
                    help="forward/backward manual-flight speed, m/s (default: utils_airsim.SPEED)")
    ap.add_argument("--z-speed", type=float, default=None,
                    help="climb/descend manual-flight speed, m/s (default: utils_airsim.Z_SPEED)")
    ap.add_argument("--turn-speed", type=float, default=None,
                    help="yaw manual-flight rate, deg/s (default: utils_airsim.TURN_SPEED)")
    ap.add_argument("--depth-decode", choices=["lut", "packed"], default="lut",
                    help="how to convert AirSim's non-float DepthPerspective capture to "
                         "metric meters. 'lut' (default): empirically calibrate an "
                         "8-bit-quantized lookup table against a one-time float capture per "
                         "--out (see module docstring - needs a lucky float capture that can "
                         "hang). 'packed': decode every frame's 3 channels directly as a "
                         "24-bit millimeter integer (ported from airsim_recorder.py) - no "
                         "float capture ever needed, and full mm precision instead of 256 "
                         "quantization levels, but not yet cross-validated against 'lut' on "
                         "this build; opt in and compare depth/ outputs before trusting it "
                         "for a real campaign.")
    ap.add_argument("--renderer", choices=["vulkan", "opengl4"], default=None,
                    help="force AirSim's rendering backend via UE4's -vulkan/-opengl4 flag "
                         "(default: AirSim's own default, Vulkan on Linux). Worth trying "
                         "opengl4 if float depth capture hangs its RPC - a known Vulkan-specific "
                         "issue on some Linux/driver combinations.")
    ap.add_argument("--no-preview", action="store_true",
                    help="disable the small left/right/depth preview window shown during "
                         "recording (default: shown, like airsim_recorder.py's). Useful to "
                         "shave the small per-frame cv2.imshow/waitKey cost, or to stop the "
                         "window from grabbing focus during a long unattended "
                         "run_dataset_campaign.sh session.")
    args = ap.parse_args()

    speed = args.speed if args.speed is not None else utils_airsim.SPEED
    z_speed = args.z_speed if args.z_speed is not None else utils_airsim.Z_SPEED
    turn_speed = args.turn_speed if args.turn_speed is not None else utils_airsim.TURN_SPEED

    left_cam, right_cams = load_cameras(args.settings)
    print(f"Cameras from {args.settings}: left={left_cam}, "
          + ", ".join(f"{name}={b * 100:.2f}cm" for name, b in right_cams))

    sim_process = None

    # settings_dataset.json is already fully resolved (no template tokens),
    # so just use it directly as AirSim's active settings.json.
    with open(args.settings) as f:
        resolved = f.read()
    with open(utils_airsim.JSON_SETTINGS, "w") as f:
        f.write(resolved)
    env_script = ENV_SCRIPTS[args.env] if args.env is not None else utils_airsim.ENV_SCRIPT
    print(f"Launching {args.env or '(utils_airsim.ENV_SCRIPT default)'}: {env_script}")
    if args.renderer is not None:
        print(f"Forcing renderer: -{args.renderer}")
    sim_process = utils_airsim.launch_sim(env_script, force_res=False, renderer=args.renderer)
    time.sleep(5)

    client = airsim.MultirotorClient()
    client.confirmConnection()

    if args.weather is not None:
        client.simEnableWeather(args.weather != "clear")
        if args.weather != "clear":
            weather_param = {"rain": airsim.WeatherParameter.Rain,
                              "fog": airsim.WeatherParameter.Fog,
                              "dust": airsim.WeatherParameter.Dust}[args.weather]
            client.simSetWeatherParameter(weather_param, args.weather_intensity)
        print(f"Weather: {args.weather}"
              + (f" (intensity {args.weather_intensity})" if args.weather != "clear" else ""))

    if args.wind is not None:
        if args.wind > 0:
            heading = random.uniform(0, 2 * math.pi)
            wind_vec = airsim.Vector3r(args.wind * math.cos(heading), args.wind * math.sin(heading), 0.0)
            print(f"Wind: {args.wind:.1f} m/s @ {math.degrees(heading):.0f} deg")
        else:
            wind_vec = airsim.Vector3r(0.0, 0.0, 0.0)
            print("Wind: calm")
        client.simSetWind(wind_vec)

    if args.time_of_day is not None:
        client.simSetTimeOfDay(True, start_datetime=args.time_of_day, is_start_datetime_dst=False,
                                celestial_clock_speed=1, update_interval_secs=60, move_sun=True)
        print(f"Time of day: {args.time_of_day}")

    # Build the keyboard listener BEFORE arming/takeoff: if pynput can't
    # actually receive key events here (wrong interpreter - pynput lives in
    # the project venv's system-site-packages, not a bare system/conda
    # python3 - or no X11/display access), fail loudly now rather than
    # leaving the drone armed and airborne with no way to control it.
    try:
        listener = _make_key_handlers()
        listener.start()
    except Exception as e:
        sys.exit(f"Keyboard listener failed to start ({e}). Common causes: this "
                 f"interpreter ({sys.executable}) doesn't have pynput installed - "
                 f"activate the project venv (obs_det_env) before running this script "
                 f"- or there's no working X11 DISPLAY (Wayland-only sessions are not "
                 f"supported by pynput's global listener).")

    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")
    print("Taking off...")
    client.takeoffAsync(vehicle_name="Drone1").join()
    time.sleep(1)
    if args.altitude is not None:
        print(f"Climbing to {args.altitude:.1f} m altitude band...")
        # Bounded timeout_sec: this defaults to effectively infinite, and
        # will block here forever if the target altitude collides with
        # terrain/foliage/a building and the drone can never quite
        # converge on it - starving the recording loop below (and the
        # keyboard-driven moveByVelocityBodyFrameAsync calls in it) of
        # ever running at all.
        client.moveToZAsync(-args.altitude, z_speed, timeout_sec=10, vehicle_name="Drone1").join()
    print("Keyboard flight enabled: arrows = move/yaw, PgUp/PgDn = altitude.")

    seq_dir = os.path.join(os.path.expanduser(args.out),
                           time.strftime(f"seq_%Y%m%d_%H%M%S_{args.tag}"))
    right_dirs = [f"right_{round(baseline * 1000):04d}mm" for _, baseline in right_cams]
    for sub in ("left", "depth", *right_dirs):
        os.makedirs(os.path.join(seq_dir, sub), exist_ok=True)

    conditions = {"tag": args.tag,
                  "env": args.env,
                  "weather": args.weather,
                  "weather_intensity": args.weather_intensity if args.weather not in (None, "clear") else None,
                  "time_of_day": args.time_of_day,
                  "altitude_target_m": args.altitude,
                  "wind_mps": args.wind,
                  "speed_mps": speed,
                  "z_speed_mps": z_speed,
                  "turn_speed_dps": turn_speed}
    with open(os.path.join(seq_dir, "conditions.json"), "w") as f:
        json.dump(conditions, f, indent=2)

    # Response order: left Scene, then one Scene per right camera. Used both
    # for the one-time stationary baseline calibration below and (extended
    # with a depth request) for the per-frame recording loop further down.
    requests = [
        airsim.ImageRequest(left_cam, airsim.ImageType.Scene, False, False),
    ] + [airsim.ImageRequest(name, airsim.ImageType.Scene, False, False)
         for name, _ in right_cams]

    # One-time calibration capture, BEFORE the recording loop (and thus
    # before any keyboard-driven movement command is ever sent): each Scene
    # request below is a separate sequential simGetImages call (per
    # fetch_image's docstring), so if the drone were moving while these are
    # taken, each right camera's measured baseline would include however far
    # the drone moved between captures on top of the true fixed rig offset -
    # silently corrupting every frame's disparity ground truth for the
    # whole ride. Capturing here, while still stationary post-takeoff/climb,
    # avoids that entirely.
    calib_responses = [fetch_image(client, r) for r in requests]
    calib_left = decode_rgb(calib_responses[0])
    if calib_left is None:
        sys.exit("Calibration capture failed: left camera Scene response didn't decode.")
    h, w = calib_left.shape[:2]
    fx = (w / 2.0) / math.tan(math.radians(FOV_DEGREES) / 2.0)
    # DepthPerspective is the Euclidean range R from the camera to each
    # point; the disp = fx*B/depth formula (and everything
    # downstream) assumes planar depth Z (distance along the optical axis)
    # instead. Standard pinhole relation: R = Z * sqrt(1 + dx^2 + dy^2)
    # where dx,dy are the pixel's normalized offset from the principal
    # point -> Z = R / that factor.
    du, dv = np.meshgrid((np.arange(w) - w / 2.0) / fx, (np.arange(h) - h / 2.0) / fx)
    depth_correction = 1.0 / np.sqrt(1.0 + du ** 2 + dv ** 2)
    pl = calib_responses[0].camera_position
    pl = (pl.x_val, pl.y_val, pl.z_val)
    stereo_pairs = []
    for (name, _nominal), resp, dir_name in zip(right_cams, calib_responses[1:], right_dirs):
        pr = resp.camera_position
        baseline = math.dist(pl, (pr.x_val, pr.y_val, pr.z_val))
        stereo_pairs.append({"camera": name, "dir": dir_name,
                             "baseline_m": baseline,
                             "fx_baseline_disp_at_1m": fx * baseline})
    calib = {"width": w, "height": h, "fov_degrees": FOV_DEGREES,
             "fx": fx, "fy": fx, "cx": w / 2.0, "cy": h / 2.0,
             "fps": args.fps, "depth_zfar_m": 100.0,
             "left_camera": left_cam,
             "stereo_pairs": stereo_pairs,
             "pose_convention": "left camera pose in world NED; camera frame "
                                "x=forward y=right z=down (AirSim). Convert to "
                                "optical (x=right y=down z=fwd) in the loader."}
    with open(os.path.join(seq_dir, "calib.json"), "w") as f:
        json.dump(calib, f, indent=2)
    print(f"calib: {w}x{h}, fx={fx:.1f}px, "
          + ", ".join(f"{p['dir']}={p['baseline_m']:.4f}m" for p in stereo_pairs))

    # Per-frame requests: the calibration Scene set above, plus one non-float
    # depth request - fetched sequentially like everything else (fetch_image),
    # on this same single connection.
    depth_request = airsim.ImageRequest(left_cam, airsim.ImageType.DepthPerspective, False, False)
    frame_requests = requests + [depth_request]
    n_expected = len(frame_requests)

    depth_lut = None
    packed_order = None
    if args.depth_decode == "lut":
        # depth_lut.npy converts the per-frame non-float depth capture below
        # to metric meters (see module docstring + build_depth_lut). Shared
        # per --out dir: once any ride successfully bootstraps it, every
        # later ride writing to the same --out reuses it without needing its
        # own lucky float capture.
        depth_lut_path = os.path.join(os.path.expanduser(args.out), "depth_lut.npy")
        if os.path.exists(depth_lut_path):
            depth_lut = np.load(depth_lut_path)
            print(f"  Loaded existing depth calibration: {depth_lut_path}")
        else:
            print(f"  No depth calibration yet for {os.path.expanduser(args.out)} - attempting "
                  f"a one-time paired float+non-float capture...")
            depth_lut = try_calibrate_depth_lut(client, left_cam)
            if depth_lut is not None:
                np.save(depth_lut_path, depth_lut)
                print(f"  Calibration succeeded -> saved {depth_lut_path} (every ride writing "
                      f"to this --out will reuse it from now on).")
            else:
                print(f"  [warn] calibration failed (float capture didn't decode) - no depth "
                      f"data possible for this ride. A later ride may still bootstrap it.")
        depth_ready = depth_lut is not None
    else:
        # packed mode needs no float capture at all (see depth_packed_to_meters) -
        # just one non-float depth frame, still stationary post-takeoff/climb
        # like the calibration captures above, to auto-detect the channel order.
        order_resp = fetch_image(client, depth_request)
        order_img = decode_depth_raw_uint8(order_resp)
        if order_img is None:
            sys.exit("Packed depth auto-detect capture failed: DepthPerspective response "
                      "didn't decode.")
        packed_order = pick_packed_order(order_img)
        print(f"  Packed depth channel order detected: {packed_order.upper()}")
        depth_ready = True

    period = 1.0 / args.fps
    # At least 1s, not just 2x the intended loop period: each loop iteration
    # here fetches 4 images at 640x480 (~3.7 MB, one combined simGetImages
    # call) AND writes 3 PNGs + a depth .npy to disk, unlike
    # airsim_demo.py/publish_airsim.py's much lighter 2x 448x256 RGB-only
    # loop. If a slow iteration ever takes longer than cmd_duration, the
    # velocity command expires and the drone reverts to hover before the
    # next command is sent - a real way to get "barely moves" without
    # anything being wrong with the keyboard input.
    cmd_duration = max(2.0 * period, 1.0)

    # Fixed, non-interactive countdown (not a keypress prompt, so it doesn't
    # block the orchestrator's automation) rather than an instant start:
    # --duration is short (as low as 6s in run_dataset_campaign.sh) and the
    # recording clock below starts right after this, so without a moment to
    # get your hands on the keyboard first, a ride can end before you've
    # pressed anything at all. Flight keys are read and applied throughout
    # (not just during the recording loop), so you can already be moving
    # when recording starts instead of starting from a dead stop.
    countdown_sec = 3
    next_tick = time.monotonic() + 1
    remaining = countdown_sec
    print(f"  recording starts in {remaining}... (you can fly now)")
    countdown_end = time.monotonic() + countdown_sec
    while time.monotonic() < countdown_end:
        send_flight_command(client, speed, z_speed, turn_speed, cmd_duration)
        if time.monotonic() >= next_tick:
            remaining -= 1
            next_tick += 1
            if remaining > 0:
                print(f"  recording starts in {remaining}...")
        time.sleep(0.1)

    if not args.no_preview:
        cv2.namedWindow("collect_dataset.py preview", cv2.WINDOW_NORMAL)

    # Bounded (not unbounded) so a disk that can't keep up applies
    # backpressure via a blocking queue.put() instead of growing memory
    # without limit - see save_worker's docstring for why disk I/O, unlike
    # the AirSim RPC fetch, is safe to hand to a second thread.
    save_queue = queue.Queue(maxsize=32)
    write_stats = {"n": 0, "seconds": 0.0}
    save_thread = threading.Thread(target=save_worker, args=(save_queue, write_stats), daemon=True)
    save_thread.start()

    pose_file = open(os.path.join(seq_dir, "poses.csv"), "w", newline="")
    pose_csv = csv.writer(pose_file)
    pose_csv.writerow(["idx", "timestamp_ns",
                       "cam_px", "cam_py", "cam_pz",
                       "cam_qw", "cam_qx", "cam_qy", "cam_qz",
                       "body_vx", "body_vy", "body_vz",
                       "body_wx", "body_wy", "body_wz"])

    idx = 0
    dropped = 0
    last_pos = None
    t_start = time.monotonic()
    print(f"Recording to {seq_dir} for {args.duration:.0f}s at {args.fps:.0f} FPS. Ctrl+C to stop early.")

    # Per-stage timing accumulators, reported in the "Done:" summary below -
    # not for permanent profiling, just to see WHERE a slow-FPS ride's time
    # actually goes instead of guessing. n_timed counts iterations that
    # reached the fetch stage (i.e., excludes the no-depth-calibration skip
    # above). sum_save is now just the save_queue.put() handoff cost (real
    # disk-write time moved to save_worker's thread - see write_stats for
    # that) - it should stay ~0 unless the writer can't keep up and the
    # bounded queue's backpressure starts blocking put(), which would still
    # show up here as exactly the same FPS-capping cost it used to be.
    sum_fetch = sum_decode = sum_preview = sum_save = 0.0
    n_timed = 0

    try:
        while time.monotonic() - t_start < args.duration:
            t_loop = time.monotonic()

            send_flight_command(client, speed, z_speed, turn_speed, cmd_duration)

            if not depth_ready:
                dropped += 1
                if dropped <= 5 or dropped % 20 == 0:
                    print(f"  [warn] DROPPED (no depth calibration available for this --out) "
                          f"-> total dropped so far: {dropped} (saved so far: {idx})")
                time.sleep(period)  # avoid a tight CPU spin for the rest of this ride
                continue

            # One COMBINED simGetImages call for all of Scene x N + non-float
            # depth. Requests used to be split one-per-call because
            # combining *float* depth capture with anything else could hang
            # AirSim's RPC indefinitely (see module docstring) - but depth
            # is non-float only now, which has been reliable in every test,
            # including combined with Scene requests (airsim_recorder.py's
            # own working code does exactly this). Splitting them cost ~5
            # sequential RPC round-trips per frame for no remaining safety
            # benefit, capping throughput well under the requested --fps.
            t_fetch0 = time.monotonic()
            responses = client.simGetImages(frame_requests)
            state = client.getMultirotorState(vehicle_name="Drone1")
            t_fetch1 = time.monotonic()
            sum_fetch += t_fetch1 - t_fetch0
            n_timed += 1

            if len(responses) != n_expected:
                dropped += 1
                if dropped <= 5 or dropped % 20 == 0:
                    print(f"  [warn] DROPPED (response count): got {len(responses)}/{n_expected} "
                          f"-> total dropped so far: {dropped} (saved so far: {idx})")
                continue
            left = decode_rgb(responses[0])
            rights = [decode_rgb(r) for r in responses[1:-1]]
            if args.depth_decode == "lut":
                depth_raw = decode_depth_nonfloat(responses[-1])
            else:
                depth_raw = decode_depth_raw_uint8(responses[-1])
            if left is None or depth_raw is None or any(r is None for r in rights):
                dropped += 1
                if dropped <= 5 or dropped % 20 == 0:
                    print(f"  [warn] DROPPED (decode failure): left_ok={left is not None} "
                          f"depth_ok={depth_raw is not None} "
                          f"rights_ok={[r is not None for r in rights]} "
                          f"-> total dropped so far: {dropped} (saved so far: {idx})")
                continue
            if args.depth_decode == "lut":
                depth = (depth_lut[depth_raw] * depth_correction).astype(np.float32)
            else:
                depth = (depth_packed_to_meters(depth_raw, packed_order) * depth_correction).astype(np.float32)
            t_decode1 = time.monotonic()
            sum_decode += t_decode1 - t_fetch1

            if not args.no_preview:
                # Small sanity-check collage, not a real viewer - left RGB,
                # first right RGB (if any), and depth colorized against a
                # PER-FRAME dynamic range, all shrunk to PREVIEW_HEIGHT_PX
                # tall. Modeled on airsim_recorder.py's preview window,
                # minus its 2x2 layout (kept to one row here since "small"
                # was the point) and minus its FIXED normalization range -
                # a fixed ceiling (this used to be PREVIEW_MAX_DEPTH_M=30m,
                # matching airsim_recorder.py's own hardcoded 10m) silently
                # clips to one flat color whenever the whole frame is
                # farther away than that (observed: a scene at ~60-66m
                # against a 30m ceiling clips 100% of pixels to the same
                # saturated color even though the underlying depth varies
                # fine - a visualization bug, not a decode bug). 2nd/98th
                # percentiles (not raw min/max) avoid a handful of noisy
                # outlier pixels stretching the whole scale uselessly.
                # Depth is downscaled BEFORE colorizing (not after) so
                # applyColorMap runs on ~5x fewer pixels - colorizing first
                # and shrinking the result after was needlessly doing
                # full-resolution work for a 200px-tall preview.
                scale = PREVIEW_HEIGHT_PX / left.shape[0]
                pw = int(left.shape[1] * scale)
                depth_small = cv2.resize(depth, (pw, PREVIEW_HEIGHT_PX), interpolation=cv2.INTER_NEAREST)
                # SQRT-scale normalization, not linear (and not log
                # either - log was tried first and overcorrected: it
                # stretched near-field contrast so aggressively that
                # far-field depths got squashed into one flat bright blob
                # with no shading left among themselves). sqrt is a gentler
                # compression - for a 3-90m scene it gives near (3-15m) and
                # far (60-90m) roughly comparable shares of the color range
                # instead of one dominating the other, which is the balance
                # PREVIEW_GAMMA exists to retune if this still isn't right.
                depth_pow = np.power(np.maximum(depth_small, 0.0), PREVIEW_GAMMA)
                valid = (depth_small > 0.05) & (depth_small < 99.5)
                if np.any(valid):
                    d_lo, d_hi = np.percentile(depth_pow[valid], [2, 98])
                    if d_hi <= d_lo:
                        d_hi = d_lo + 1.0
                else:
                    d_lo, d_hi = 0.05 ** PREVIEW_GAMMA, PREVIEW_MAX_DEPTH_M ** PREVIEW_GAMMA
                depth_cm = cv2.applyColorMap(
                    (np.clip((depth_pow - d_lo) / (d_hi - d_lo), 0, 1) * 255).astype(np.uint8),
                    cv2.COLORMAP_MAGMA)
                rgb_panels = [cv2.resize(p, (pw, PREVIEW_HEIGHT_PX))
                              for p in [left] + (rights[:1] if rights else [])]
                collage = np.hstack(rgb_panels + [depth_cm])
                cv2.putText(collage, f"saved {idx}  dropped {dropped}", (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # Show the actual per-frame range the colormap above is
                # normalized against (d_lo-d_hi, converted back out of
                # depth**PREVIEW_GAMMA space to meters for readability),
                # not raw min/max - this is what to check first if the
                # preview ever looks flat again.
                cv2.putText(collage,
                            f"depth {d_lo ** (1 / PREVIEW_GAMMA):.1f}-{d_hi ** (1 / PREVIEW_GAMMA):.1f}m "
                            f"(full {depth.min():.1f}-{depth.max():.1f}m)",
                            (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.imshow("collect_dataset.py preview", collage)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("\nESC pressed - stopping recording early.")
                    break
            t_preview1 = time.monotonic()
            sum_preview += t_preview1 - t_decode1

            cam_p = responses[0].camera_position
            if args.min_motion > 0 and last_pos is not None:
                if math.dist((cam_p.x_val, cam_p.y_val, cam_p.z_val), last_pos) < args.min_motion:
                    continue
            last_pos = (cam_p.x_val, cam_p.y_val, cam_p.z_val)

            name = f"{idx:06d}"
            # Handoff only - the actual cv2.imwrite/np.save calls happen on
            # save_worker's thread now, off the fetch-bound critical path.
            # left/depth/rights are fresh arrays this iteration (decode_rgb/
            # depth_packed_to_meters/the LUT lookup all allocate new ones
            # each call), so there's no aliasing risk in handing them to
            # another thread instead of writing them here.
            save_queue.put((seq_dir, name, left, depth, rights, right_dirs))
            sum_save += time.monotonic() - t_preview1

            cam_q = responses[0].camera_orientation
            kin = state.kinematics_estimated
            pose_csv.writerow([idx, responses[0].time_stamp,
                               cam_p.x_val, cam_p.y_val, cam_p.z_val,
                               cam_q.w_val, cam_q.x_val, cam_q.y_val, cam_q.z_val,
                               kin.linear_velocity.x_val, kin.linear_velocity.y_val,
                               kin.linear_velocity.z_val,
                               kin.angular_velocity.x_val, kin.angular_velocity.y_val,
                               kin.angular_velocity.z_val])
            idx += 1
            if idx == 1 or idx % 100 == 0:
                # Depth min/mean/max sanity check - printed for the very
                # first saved frame (not just every 100th, which a short
                # test ride may never reach) so a bad decode (e.g. a
                # near-flat range) is visible immediately instead of only
                # after inspecting saved depth/*.npy files by hand.
                valid_depth = depth[(depth > 0) & (depth < 100)]
                depth_stats = (f"depth min/mean/max="
                               f"{valid_depth.min():.1f}/{valid_depth.mean():.1f}/{valid_depth.max():.1f}m"
                               if valid_depth.size else "depth: no valid (0-100m) pixels")
                elapsed = time.monotonic() - t_start
                print(f"  {idx} frames, {elapsed:.0f}s elapsed, "
                      f"effective {idx / elapsed:.1f} FPS, dropped {dropped}, {depth_stats}")

            sleep = period - (time.monotonic() - t_loop)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        pose_file.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
        listener.stop()
        try:
            client.hoverAsync(vehicle_name="Drone1").join()
        except Exception:
            pass
        elapsed = time.monotonic() - t_start
        print(f"Done: {idx} frames in {elapsed:.0f}s ({idx / max(elapsed, 1e-6):.1f} FPS), "
              f"{dropped} dropped -> {seq_dir}")
        if n_timed:
            print(f"  avg per-frame: fetch(RPC)={1000 * sum_fetch / n_timed:.1f}ms "
                  f"decode={1000 * sum_decode / n_timed:.1f}ms "
                  f"preview={1000 * sum_preview / n_timed:.1f}ms "
                  f"save(enqueue)={1000 * sum_save / n_timed:.1f}ms "
                  f"(target period={1000 * period:.1f}ms) - use this to see which stage is "
                  f"actually capping FPS below --fps.")
        # Flush the background writer AFTER computing the FPS numbers above,
        # so waiting for straggling disk writes doesn't inflate `elapsed`
        # and understate the real capture throughput. daemon=True on
        # save_thread means an unflushed queue would otherwise be silently
        # dropped at interpreter exit - this join makes sure every enqueued
        # frame actually lands on disk before the process ends.
        save_queue.put(None)
        flush_start = time.monotonic()
        save_queue.join()
        if write_stats["n"]:
            print(f"  background writer: {write_stats['n']} frames written, avg "
                  f"{1000 * write_stats['seconds'] / write_stats['n']:.1f}ms/frame actual disk "
                  f"write (off the capture critical path); "
                  f"{time.monotonic() - flush_start:.1f}s to flush the remaining queue.")
        if sim_process is not None and sim_process.poll() is None:
            print("Leaving the simulator running (kill it manually if you're done).")


if __name__ == "__main__":
    main()
