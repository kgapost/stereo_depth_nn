"""Standalone diagnostic: isolate which simGetImages variant hangs.
Run against an already-running AirSim instance (no takeoff/flight needed).

Each test uses its OWN fresh MultirotorClient connection and runs the
request in a daemon thread, joined with a timeout - unlike a signal-based
timeout, this can't corrupt a mid-parse msgpack stream, so one hanging test
can't poison the ones after it.

Usage (with the project venv active):
    python debug_depth_capture.py
"""
import threading
import time

import airsim

tests = [
    ("Scene uint8 (known-working baseline)",
     airsim.ImageRequest("Camera1", airsim.ImageType.Scene, False, False)),
    ("DepthPlanar float (collect_dataset.py's current request)",
     airsim.ImageRequest("Camera1", airsim.ImageType.DepthPlanar, True, False)),
    ("DepthPlanar non-float (airsim_demo.py's working request)",
     airsim.ImageRequest("Camera1", airsim.ImageType.DepthPlanar, False, False)),
    ("DepthPerspective float",
     airsim.ImageRequest("Camera1", airsim.ImageType.DepthPerspective, True, False)),
    ("DepthPerspective non-float",
     airsim.ImageRequest("Camera1", airsim.ImageType.DepthPerspective, False, False)),
]

TIMEOUT_S = 15


def _run_request(client, req, result):
    try:
        result["resp"] = client.simGetImages([req])
    except Exception as e:
        result["exc"] = e


for label, req in tests:
    print(f"Testing: {label} ...", flush=True)

    client = airsim.MultirotorClient()
    client.confirmConnection()

    result = {}
    t0 = time.monotonic()
    thread = threading.Thread(target=_run_request, args=(client, req, result), daemon=True)
    thread.start()
    thread.join(timeout=TIMEOUT_S)
    elapsed = time.monotonic() - t0

    if thread.is_alive():
        print(f"  TIMED OUT after {TIMEOUT_S}s (hung) - abandoning this connection, "
              f"moving on with a fresh one")
    elif "exc" in result:
        print(f"  EXCEPTION after {elapsed:.2f}s: {result['exc']}")
    else:
        resp = result.get("resp")
        if resp:
            r = resp[0]
            print(f"  OK in {elapsed:.2f}s: {r.width}x{r.height}, "
                  f"uint8_bytes={len(r.image_data_uint8)}, "
                  f"float_len={len(r.image_data_float)}")
        else:
            print(f"  got EMPTY response list in {elapsed:.2f}s")
    print()

print("Done testing all variants.")
