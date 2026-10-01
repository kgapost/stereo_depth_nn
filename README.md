# Stereo Depth NN

Research code that trains a neural network to guess **depth** (how far away things are) from a **stereo camera pair** (two cameras, side by side, like human eyes).

This is for a drone project. The drone needs to know how far obstacles are, so it can avoid them. The full drone project lives in a separate repo, [obstacle_detection_sim](https://github.com/kgapost/obstacle-detection-sim). This repo is only for the AI research part: training and testing new depth models. It is not needed to fly the drone.

This README has two parts:
- **Quick Start** - how to set up this repo and run the code (collect data, train, evaluate).
- **Full research documentation** - everything else: the problem this solves, the data, prior work, every model tried, the loss functions, and the training/evaluation results. This part used to be a separate file, `PAPER.md` - it is now all here instead, so there is only one document to keep up to date.

## What is in this repo

| File / folder | What it does |
|---|---|
| `collect_dataset.py` | Records left camera, right camera, and the true depth from AirSim. Saves them to disk as a dataset. |
| `debug_depth_capture.py` | A small test tool. Checks which AirSim depth-capture method works, without needing to fly. |
| `run_dataset_campaign.sh` | Runs `collect_dataset.py` many times in a row - once per world / weather / speed combination. |
| `stereo_datasets.py` | Reads a saved dataset (or the public TartanAir dataset) and prepares it for training. |
| `models/` | The neural network models themselves (see the model table below). |
| `losses.py` | The different training loss functions you can pick with `--loss`. |
| `train.py` | Trains one model on a dataset you collected with `collect_dataset.py`. |
| `train_tartanair.py` | Trains one model (or many, in a "grid search") on the public TartanAir dataset. |
| `evaluate.py` | Checks how good a trained model is, and measures how fast it runs. |
| `eval_all.py` | Batch version of `evaluate.py` for a whole grid-search output folder: health-checks every run (did it finish, does it have a saved model), re-evaluates each one on two held-out data splits, measures inference speed, and flags over/underfitting - one CSV row per run. |
| `config.py`, `utils_airsim.py`, `json_templates/` | Copied from the drone repo. These start AirSim and read its camera settings. Needed only by `collect_dataset.py` and `debug_depth_capture.py` - the training/evaluation files above never touch them. |

## Setup

**1. AirSim must already be installed and able to run.** This repo does not install AirSim - it only talks to it. For the full install steps (Unreal Engine, AirSim, the AirSim worlds used below), see [obstacle_detection_sim](https://github.com/kgapost/obstacle-detection-sim)'s README.md, sections 3.1 and 2.3.1.

**2. Create a Python environment just for this repo:**
```bash
python3 -m venv stereo_env
./stereo_env/bin/pip install -r requirements.txt
```
This is a **separate, smaller** environment than the drone project's. It only needs PyTorch, OpenCV, NumPy, and (for the `yolo2d_3dhg` model only) `ultralytics` - no ROS.

**3. Point at your AirSim binaries, if needed.** `utils_airsim.py` looks for an `AirSim` folder next to this repo by default. If your AirSim binaries live somewhere else, tell it where:
```bash
export AIRSIM_BINARY_ROOT=/path/to/the/folder/that/contains/AirSimNH
```

## Step 1 - Collect a dataset from AirSim

Start AirSim yourself first, or let `collect_dataset.py` launch it for you:

```bash
# AirSim is already running:
python collect_dataset.py --out ~/datasets/airsim_stereo --tag neighborhood

# or let this script launch AirSim itself:
python collect_dataset.py --out ~/datasets/airsim_stereo --tag neighborhood --launch
```

Fly the drone with the arrow keys (move / turn) and PageUp / PageDown (go up / down). Recording stops after `--duration` seconds (10 by default) or when you press Ctrl+C.

The camera setup (how many cameras, how far apart) comes from `json_templates/settings_dataset.json`. Edit that file to change the cameras - you do not need to change any Python code.

To record many short rides automatically, across different AirSim worlds:
```bash
DRY_RUN=1 ./run_dataset_campaign.sh ~/datasets/airsim_stereo   # print the plan first, run nothing
./run_dataset_campaign.sh ~/datasets/airsim_stereo
```

More detail on all of this - which AirSim worlds are used and why, exactly what gets recorded, file sizes, and the ground-truth math - is in [Section 2, Dataset Creation](#2-dataset-creation) below.

## Step 2 - Train a model

Twelve `--model` choices are available: eight base architectures (Sections 6-11), four of which also have an `fxb`-conditioned sibling variant that additionally takes the camera's focal-length x baseline as an input, so one trained model generalises across whatever stereo rig it's given at inference (Section 5 explains which four, and why the other four don't need one). All of them are already built and working - none are just ideas on paper. The [Model Summary](#13-model-summary) section below has the full table with parameter counts; [Sections 6-11](#6-proposed-method---slow-baseline-approach-stereoconvnet) explain how each base architecture actually works.

Train one model on data you collected in Step 1:
```bash
python train.py --model siam2d_3dhg --data ~/datasets/airsim_stereo --out runs/siam2d_3dhg
```

Or train on the public TartanAir dataset instead:
```bash
python train_tartanair.py --model siam2d_egomotion_fxb --data ~/datasets/tartanair --out runs/siam2d_egomotion_fxb
```

Both scripts take many more options, for example `--loss` (which training objective to use, see [Section 14, Loss Functions](#14-loss-functions)), `--bs` (batch size), and `--epochs`. To see the full list:
```bash
python train.py --help
python train_tartanair.py --help
```
The complete argument reference is also written out in [Section 15.4](#154-command-line-reference-train_tartanairpy).

### Try many settings at once (grid search)

`train_tartanair.py --grid-search` trains many combinations of model, batch size, learning rate, and loss function automatically, and writes one `summary.csv` file comparing all of them:

```bash
python train_tartanair.py --grid-search --data ~/datasets/tartanair \
    --grid-models siam2d_3dhg siam2d_egomotion_fxb mobile2d_3dhg pyr2d_casc2d \
    --out runs/my_grid_search
```

Each combination trains as its own process, so one crashing (for example, running out of GPU memory) does not stop the others - it is just marked `FAILED` in the summary. If the sweep is stopped (Ctrl+C, or it crashes) and started again with the same command, finished combinations are skipped instead of redone. Full detail, including measured GPU memory per model and the results of the sweeps run so far, is in [Section 15.3](#153-grid-search).

## Step 3 - Check how good a trained model is

```bash
python evaluate.py --ckpt runs/siam2d_3dhg/best.pth --model siam2d_3dhg --data ~/datasets/airsim_stereo_test
```

This reports how far off the depth guess is - in pixels and in metres, split by distance (close / medium / far, since the same pixel error means a much bigger real-world error far away than it does up close). It can also measure how fast the model runs, e.g. on a Jetson:
```bash
python evaluate.py --model siam2d_3dhg --bench --size 256x448 --fp16
```
Full detail on what is measured and in what order is in [Section 15.2](#152-evaluation).

## A note on camera settings

The real drone's camera is a Waveshare/Seeed IMX219-83, with a **6 cm** distance between its two lenses (the "baseline") and an **83°** field of view. `json_templates/settings_dataset.json` is already set up to match this, so depth recorded in AirSim lines up with what the real camera would see. If you change the real camera, update that file to match. See [Section 2.5](#25-disparity-ground-truth-and-loss-masking) for exactly what this baseline means for accuracy at different distances.

---

**Everything below this line is the project's full research write-up:** the problem being solved, how the training data is made, the papers this work is positioned against, a literature review, the gap in prior work, the proposed method (one section per model), the loss functions, and how training/evaluation are run - in full detail, not summarized. It used to live in a separate `PAPER.md` file; it is now part of this README instead.

## Depth from Stereo: Problem Definition, Literature Review, and Proposed Method

Problem defintion (Section 1), 
How the training data is produced (Section 2), 
Current state-of-the-art paper this work positions against (Section 3), 
A literature review (Section 4), 
The resulting gap in prior work (Section 5), 
Proposed method (Section 6, Section 7, Section 8, Section 9, Section 10, Section 11, Section 12)
Loss functions (Section 14), 
Trained and evaluation (Section 15).

---

### Table of Contents
1. [Problem Definition](#1-problem-definition)
2. [Dataset Creation](#2-dataset-creation)
3. [SoA Paper](#3-soa-paper)
4. [Related Work](#4-related-work)
5. [Gap in the Literature](#5-gap-in-the-literature)
6. [Proposed Method - Slow Baseline Approach (StereoConvNet)](#6-proposed-method---slow-baseline-approach-stereoconvnet)
7. [Proposed Method - Baseline Approach (FastStereoNet)](#7-proposed-method---baseline-approach-faststereonet)
8. [Proposed Method - Baseline Approach with Temporal Data (StereoConv3DNet)](#8-proposed-method---baseline-approach-with-temporal-data-stereoconv3dnet)
9. [Proposed Method - Baseline Fast Approach with Temporal Data (StereoConv3DNet, T=3)](#9-proposed-method---baseline-fast-approach-with-temporal-data-stereoconv3dnet-t3)
10. [Single-Frame Model Variants](#10-single-frame-model-variants)
11. [TempoBandNet](#11-tempobandnet)
12. [YOLO Techniques Used](#12-yolo-techniques-used)
13. [Model Summary](#13-model-summary)
14. [Loss Functions](#14-loss-functions)
15. [Training and Evaluation](#15-training-and-evaluation)
16. [Bibliography](#16-bibliography)


### 1. Problem Definition

**Goal in one line**: train a single model that takes a stereo image pair
(plus the camera's baseline distance) and directly outputs a dense depth map,
in real-world units (metres), that stays stable from frame to frame.

Right now, the project cleans up its depth map with a separate step after
the fact (e.g. WLS filtering), because the raw output is noisy and has holes
in it. The goal here is to replace that hand-tuned clean-up with a single
trained network that already gives a clean result.

#### 1.1 Requirements

| # | Requirement | Why |
|---|---|---|
| 1 | **Fast**: real time on a small onboard computer (Jetson-class) | The whole point is to run this on a drone |
| 2 | **Accurate, in real units (metres)** | Obstacle avoidance needs real distances, not just relative ones - this is why camera-only (monocular) methods are not enough by themselves |
| 3 | **Robust to noise**, without hand-tuned filters | Needed for reliable obstacle avoidance |
| 4 | **Find and fill in** blind spots, better than simple copy-nearest-pixel | See below |
| 5 | **Use motion over time** to make the map more reliable | See below |

#### 1.2 Blind spots and shadows

The current code already shows this is a known, unsolved problem, not a new
one:

- `STEREO_MASK_INVALID` (on by default) marks every pixel the matcher
  could not match as invalid (NaN). These pixels are simply thrown away,
  not filled in, so they end up as holes in the 3D map.
- `mask_depth_shadows()` and `bilateral_filtering()` are two functions
  written for exactly this problem, but neither is ever called anywhere in
  the code - dead code left over from an earlier, abandoned attempt.
- Even switched back on, `mask_depth_shadows` only zeroes out the bad
  pixels - it does not try to fill them back in.

The problem covers three cases: spots the two cameras can't both see
(occlusions), flat, plain surfaces with nothing to match ("shadow"
regions), and shiny or repeating surfaces - all cases where block matching
comes back with no usable data at all. The requirement has two parts: (a)
find these bad regions, and (b) fill them in with something better than
copying the nearest pixel - for example, a small learned "fill-in" step
that uses the colour image and, when available, information carried over
from previous frames, instead of smearing whatever wrong value sits at the
edge of the hole.

#### 1.3 Temporal motion

The current code smooths the depth map over time with a simple running
average: `depth = alpha*current + (1-alpha)*previous`, with a safety check
that resets the average if a pixel's value jumps too far. The problem: this
averaging **does not know the camera is moving** - it assumes the pixel at
position (x, y) shows the same surface point this frame as it did last
frame. That is only true if the camera stands still. On a flying drone it
is never true, so this smoothing just trades noise for motion blur and lag
- it is not really using the history available. This project's
`TempoBandNet` model (Section 11) already fixes this by tracking the
drone's own motion and shifting the previous frame's data to where it
should be now. Section 8's model tests the plainer alternative: stack
several raw frames together and let the network figure out on its own
whether that helps, with no explicit motion tracking - a simple baseline to
compare `TempoBandNet` against.

---

### 2. Dataset Creation

All ground truth (the "correct answer" used to train the network) comes
from AirSim's per-pixel `DepthPlanar` image. This is not an estimate - it
comes straight from the renderer, so it is dense (every pixel has a value),
clean (no sensor noise), and already lined up with the colour image
(the same approach Scene Flow [[mayer2016sceneflow]](#16-bibliography) and
TartanAir [[wang2020tartanair]](#16-bibliography) use). Section 2.5 explains
exactly how the ground truth and loss masking work, shared by every model
in Section 6 through Section 11; Section 2.6 explains how to actually run
the data collection.

#### 2.1 AirSim environments

The project uses seven ready-made AirSim v1.8.1 worlds (one per entry in
`utils_airsim.ENV_SCRIPT`), each chosen to test a *different* requirement
from Section 1:

| Environment | Scene type | What it tests |
|---|---|---|
| **Blocks** | Simple shapes, sharp edges, plain surfaces | Checks that calibration and ground truth line up correctly; the easiest scene, good for debugging |
| **AirSimNH** | Suburban neighborhood - houses, hedges, parked cars | Close-range clutter typical of low-altitude flight |
| **AbandonedPark** | Park - benches, trees, paths, open lawns | Different close-range clutter than AirSimNH (uneven plants vs. straight rows of houses), same altitude |
| **LandscapeMountains** | Large natural terrain | Long-range, plain surfaces - the hardest case for accurate depth at range, and where "shadow" regions (Section 1.3) show up |
| **Africa_Savannah** | Open terrain with moving animals | Objects moving on their own at long range - tests how well the model handles motion, since the assumption "the world stays still" is broken on purpose here |
| **ZhangJiajie** | Dense plants, tall rock formations | Lots of occlusion - the best source of real blind spots, not just rendering glitches |
| **TrapCam** | Dense forest, wildlife-camera style | Moving objects again, but close-up and under heavy plant cover at the same time - the hardest combination of the two problems above |

#### 2.2 Rides, duration, and content per ride

- **30 rides x 7 environments = 210 rides**, 10 seconds each (100 frames at
  10 FPS, the `collect_dataset.py` default). Many short rides work better
  than a few long ones: back-to-back frames at 10 FPS look almost the same
  anyway, so what actually adds variety is the *number* of rides, not how
  long each one is.
- Each ride changes (and records) at least one of: weather (clear / rain /
  fog / dust), time of day, altitude band (under 5 m, 5-20 m cruise, above
  20 m long-range), and flight style (slow hover, fast forward flight,
  continuous turning). With many short rides instead of a few long ones,
  each ride can get its own combination of these settings, giving much
  better coverage than a handful of long sessions would.
- **Only one camera baseline for now**: the rig currently uses one right
  camera, at the 6.0 cm baseline of the project's real onboard camera
  (Waveshare/Seeed IMX219-83), instead of the two baselines (6.0 cm and
  9.5 cm) an earlier version of the plan recorded at once. Keeping it to
  one baseline for this first round of data collection - which everything
  in Section 6 through Section 11 builds on - keeps the focus on getting
  the capture pipeline and training recipe right before adding more
  complexity. The camera rig and scripts don't need to change to add a
  second baseline later (Section 2.3, Section 2.6): `collect_dataset.py`
  already supports any number of right cameras, set in the settings file,
  so this is a config change, not a code change - see the note at the end
  of Section 2.5 for what that unlocks.

#### 2.3 What one frame contains and how it's gathered

`collect_dataset.py` grabs every image below in a **single `simGetImages`
call**, so they all come from the exact same render and pixel positions
line up. It reads the camera setup from an AirSim settings file
([settings_dataset.json](json_templates/settings_dataset.json)) instead
of hardcoding camera names - the first camera listed is the left/reference
camera, and every other camera is a right camera at its own distance from
it:

| item | source | role |
|---|---|---|
| left RGB | `Camera1`, colour image | network input (shared reference) |
| right RGB | `Camera2`, colour image, 6.0 cm from the left camera | network input |
| left depth | `Camera1`, `DepthPlanar` (float32, metres) | ground truth |
| left camera pose | comes with the image response (world position + orientation) | motion info for the temporal models |
| body velocities | `getMultirotorState` | extra recorded info |

The camera pose is read straight from the image response, not from a
separate query, so the pose and the pixels always match up in time. Each
right camera's baseline is measured from where it actually is relative to
the left camera, rather than just trusted from the settings file.

#### 2.4 Storage format and size

```
<out>/seq_<timestamp>_<tag>/
    calib.json           fx, fy, cx, cy, width, height, fps, and one
                          stereo_pairs[] entry per right camera (camera name,
                          measured baseline_m, image dir)
    poses.csv             per-frame timestamp + left-camera pose + body velocities
    left/000000.png        left RGB
    depth/000000.npy       left DepthPlanar, float32 metres
    right_0060mm/000000.png  right RGB, 6.0 cm baseline (real onboard rig)
```

Nothing about frame order or timing is baked into the stored files: the
data loader builds training windows of any length and gap on the fly, and
works out the camera's motion between frames when needed. So one storage
format serves every model in Section 6 through Section 11 - the single-frame models
just use window length 1.

Sizing at the chosen capture resolution, **640x480** (4:3, a deliberate
step up from the 448x256 live-pipeline processing size in
[config.py](config.py), so training can crop/downsample instead of
upsampling - see the discussion that led to this choice): each frame is 1
left RGB + 1 right RGB (PNG) + 1 depth map (raw float32 `.npy`, which
takes up most of the space) - about 1.34 MB/frame. At 100 frames per ride x
210 rides that's **21,000 frames total, about 28 GB** (35 minutes of
recorded flight across all 7 environments). Adding the 9.5 cm camera back
later (Section 2.2) would only add one more small PNG per frame, since the
depth map, not the colour images, is what takes up most of the space.

This is smaller than an earlier version of the plan (90,000 frames, about
45 GB, at a lower resolution) - trading total frame count for more variety
per ride. It also changes the training advice: 21,000 raw frames is
already below Scene Flow's reference size of 35k frames
[[mayer2016sceneflow]](#16-bibliography), so the non-temporal baseline
(Section 6) should train on **every frame**, not skip some - skipping more
would leave too little data. The temporal baseline (Section 8) still uses
the full 10 FPS stream directly, as before. **Never split a single ride**
between train/val/test - always hold out whole rides, ideally a whole
environment, so near-duplicate frames (0.1 s apart) can't leak between the
training and test sets.

#### 2.5 Disparity ground truth and loss masking

Training supervises **disparity** (how many pixels a point shifts between
the left and right image), not depth directly. Disparity is the value
stereo geometry actually measures, and a given pixel error means roughly
the same thing everywhere in the image, unlike depth (see Section 14 for
why that matters for the loss function). It is computed from the recorded
depth image:

```
disp = fx · B / depth
```

At this project's capture resolution (640x480, 83° field of view, giving
`fx = 361.7 px`, scaled down from the real camera module's native
resolution) and the 6.0 cm real-rig baseline:

| baseline | disparity @ 1 m | disparity @ 10 m | disparity @ 30 m | disparity @ 95 m |
|---|---|---|---|---|
| 6.0 cm (real rig) | 21.7 px | 2.17 px | 0.72 px | 0.23 px |

This shows the sim/real mismatch already flagged in Section 1.5 in concrete
numbers: past about 22 m, this baseline gives less than one pixel of
disparity, which is a known accuracy limit of this first, single-baseline
version of the dataset, not something worked around yet. A wider baseline
(the 9.5 cm camera this plan originally recorded alongside the 6.0 cm one,
see Section 2.2) stays above one pixel out to about 34 m instead - adding
it back is the planned fix, once the single-baseline pipeline is working,
though it means the disparity value again depends on which camera pair
produced it (Section 2.4's `calib.json` already plans for this: it writes
one entry per right camera no matter how many there are).

Pixels are left out of the training loss when their depth is under 0.2 m
or over 95 m (the sky has no real depth beyond 100 m in the simulator) -
matching `MAX_DEPTH_METERS` and `OCTOMAP_MIN_DEPTH_METERS`, already used
by the live pipeline in [config.py](config.py). With one camera pair,
a disparity cutoff (e.g. `disp >= 128 px`) is just one fixed number; it
only needs to depend on which camera pair produced it once a second
baseline is back in the rig.

#### 2.6 Running the collection

`collect_dataset.py` reads the camera setup from `--settings` (default
[settings_dataset.json](json_templates/settings_dataset.json)), so
adding, removing, or repositioning cameras only means editing that file,
not the script:

```bash
# sim already running:
python collect_dataset.py --out ~/datasets/airsim_stereo --tag neighborhood
# or let it launch the binary itself:
python collect_dataset.py --out ~/datasets/airsim_stereo --tag nh --launch
```

Fly with the arrow keys (move/turn) and PageUp/PageDown (altitude) - the
same controls as `publish_airsim.py`. Each ride now runs for 10 seconds by
default (`--duration 10`) instead of 5 minutes, matching the plan in
Section 2.2; pass `--duration` to override this for a one-off longer
session. Recording stops after `--duration` seconds or on Ctrl+C;
`--min-motion` can skip saving frames where the drone is barely moving.

#### 2.7 TartanAir as the second corpus

The recorded AirSim data from 2.1-2.6 is realistic but small. **TartanAir**
[[wang2020tartanair]](#16-bibliography) adds volume: it's a large public
dataset, also made with AirSim, with dense depth and known camera poses for
every frame, so every model here (including the temporal ones) can use it.
It serves three purposes: pre-training data, a public benchmark
(Section 15.2), and the data used for the hyper-parameter sweeps in
Section 15.3. [train_tartanair.py](train_tartanair.py) trains on it, reusing
the same optimiser, schedule, loss, metrics and checkpoint format as
[train.py](train.py), so runs from the two scripts can be compared directly
- only the data differs.

**Layout.** Both released versions are found automatically by
`TartanAirDataset` ([stereo_datasets.py](stereo_datasets.py)):

```
V1 (tartanair_tools, 640x480)
  <Env>/<Easy|Hard>/P000/image_left/000000_left.png
                        /image_right/000000_right.png
                        /depth_left/000000_left_depth.npy
                        /pose_left.txt
V2 (tartanairpy, 640x640)
  <Env>/Data_<easy|hard>/P000/image_lcam_front/000000.png
                             /image_rcam_front/000000.png
                             /depth_lcam_front/000000.npy   (or *_depth.png)
                             /pose_lcam_front.txt
```

V2 stores float32 depth inside a 4-channel PNG - the raw bytes need to be
read back as numbers, not treated as a normal RGBA colour image. V2 also
ships six camera rigs; `--camera` picks one, and only `front` has been
checked against this project's pose convention. To get the data:

```bash
pip install tartanair
python -c "import tartanair as ta; ta.init('$HOME/datasets/tartanair'); \
    ta.download(env=['AbandonedFactory'], difficulty=['easy'], \
    modality=['image','depth'], camera_name=['lcam_front','rcam_front'], unzip=True)"
```

**Calibration.** Fixed by the dataset itself, not read from a per-sequence
file: no lens distortion, `fx = fy = 320` (90° field of view), 0.25 m
baseline, depth given directly in metres. Poses are position + rotation in
the same convention (`world_T_optical`) this project's own pipeline uses.

**Validation is held out by environment, not by sequence.** This is the one
place the TartanAir trainer works differently from `train.py`. Different
trajectories inside the same TartanAir environment reuse the same scenery,
so splitting by trajectory leaks information - the model would already
have seen the validation scene from another angle. `environment_split()`
instead holds out entire environments (`--val-envs`, or the last
`--val-frac` of them), and only falls back to splitting by sequence if just
one environment is available.

**The mismatch with the real camera must be kept in mind for any result
from TartanAir.** Its 0.25 m baseline gives about 80 px·m of "disparity
budget", against roughly 23 px·m for this project's 6.3 cm rig - TartanAir
disparities are about 3.5x larger at the same depth. The network only ever
sees images, so it learns whatever disparity numbers it is shown. `--scale`
shrinks the images, which shrinks disparity by the same amount:
`--scale 0.29` brings TartanAir's numbers down to match this project's rig,
at the cost of resolution. Leaving `--scale 1` is the right choice when
TartanAir is only used for pre-training or as a public benchmark, rather
than standing in for the real camera. Either way, `describe_calibration()`
prints the disparity the model is about to be trained on at 2 / 10 / 30 m,
and the depth below which `--max-disp` cuts off, at the start of every run.

The intended pipeline is pre-train here, fine-tune on the rig:

```bash
python train_tartanair.py --model siam2d_3dhg --data ~/datasets/tartanair \
    --out runs/tartanair_pre
python train.py --model siam2d_3dhg --data ~/datasets/airsim_stereo \
    --out runs/finetune --init runs/tartanair_pre/best.pth
```

#### 2.8 Testing `fxb`-conditioning across multiple baselines

Four of this project's models take `fxb` (focal length x baseline) as a
direct network input instead of only using it at the very last step to turn
disparity into depth (Section 5's model table: `FastStereoNet`,
`MobileStereoNet`, `AnyStereoNet`, `YoloStereoNet`, each alongside a plain
sibling that doesn't; the other four models were built `fxb`-conditioned
from the start with no plain sibling to compare against). The whole point of
feeding it in is that **one trained model should stay accurate when handed a
stereo pair from a rig it never trained on**, as long as it's told that
rig's `fxb` - not just an accurate model for the one baseline it happened to
train on. That claim has never actually been tested yet: every sweep run so
far (Section 15.3) trains and validates a given model on a *single* fixed
baseline (TartanAir's 0.25 m, Section 2.7, or `settings_dataset.json`'s 6 cm
real rig, Section 2.2). Testing the actual claim needs the same scene, at
the same instant, observed through **two or more genuinely different**
physical baselines, so the only thing that changes between them is the
disparity geometry - not the scene, not the timing, not anything else.

**Does TartanAir have this?** No. Both versions hardcode a single, fixed
0.25 m baseline for every trajectory and every one of V2's six camera
directions (`CALIB_V1`/`CALIB_V2` in
[stereo_datasets.py](stereo_datasets.py); confirmed independently by
TartanAir's own paper, which describes twelve synchronized cameras as six
*directions* each getting one stereo pair, all sharing that same 0.25 m
separation [[wang2020tartanair]](#16-bibliography)) - not six different
baselines. `--scale` (Section 2.7) cannot stand in for a real second
baseline either, even though shrinking the images changes the disparity
numbers by exactly the same factor a different baseline would
(`disp = fx*b/depth`, and scaling `fx` by `s` has the identical effect on
that formula as scaling `b` by `s`): it produces the same scene rendered
smaller, with the same occlusions and the same matching difficulty at every
depth, just re-labelled. A genuinely different baseline changes *which*
points are occluded and how hard they are to match, which is exactly the
part of the problem `fxb`-conditioning is supposed to help the network
adapt to. So `--scale` can check that a model doesn't break when handed an
unfamiliar `fxb` *number*, but it cannot test whether the model is actually
robust to the different geometry a real second camera produces.

**Could AirSim give us this instead, with more cameras?** Yes - and most of
the machinery for it is already sitting in this repo, unused.
`collect_dataset.py`'s `load_cameras()` already treats every camera after
the first as its own independent right camera, at whatever baseline its
`X`/`Y`/`Z` offset in the settings file implies, and captures all of them
together in the same `simGetImages` call every frame (Section 2.3) - so
every right camera sees the exact same scene at the exact same instant, the
one property this test actually needs.
[settings_dataset.json](json_templates/settings_dataset.json) already
defines **four** right cameras, not just this project's one real rig, each
commented with a real device it corresponds to:

| Camera | Baseline | Modelled after |
|---|---|---|
| `Camera2` | 6.0 cm | This project's real onboard rig (Waveshare/Seeed IMX219-83) |
| `Camera3` | 9.5 cm | Intel RealSense D455 |
| `Camera4` | 12 cm | StereoLabs ZED / ZED 2, FLIR Bumblebee2 |
| `Camera5` | 24 cm | FLIR Bumblebee XB3 (wide-baseline mode) |

Recording a ride with this settings file as-is already produces four
baselines of the same ride, written to their own
`right_0060mm/`/`right_0095mm/`/`right_0120mm/`/`right_0240mm/` folders,
with `calib.json`'s `stereo_pairs[]` already recording each one's measured
baseline and folder name (Section 2.4) - no changes to `collect_dataset.py`
at all, just running it.

**The loader-side gap this used to have is now fixed.**
`AirSimStereoDataset` ([stereo_datasets.py](stereo_datasets.py)) used to read
a single hardcoded `"right"` folder and expect a flat `calib["baseline_m"]` -
neither of which the calib.json format above actually produces (it writes
baseline-suffixed folder names and nests each one's baseline inside
`stereo_pairs[]`, not at the top level), so training against genuinely
current `collect_dataset.py` output would fail before this test could even
start. It now takes a `camera` argument - exposed as `--camera` on both
`train.py` and `evaluate.py`, the same flag name `TartanAirDataset` already
uses to pick one of TartanAir's six directional rigs (Section 2.7) - that
picks which recorded right camera/baseline a run trains or evaluates
against, by AirSim camera name (`Camera3`) or image folder (`right_0095mm`);
left unset, it defaults to the first `stereo_pairs` entry, so existing
single-right-camera recordings need no flag at all.

**The test itself**, now that the loader supports it:

1. Record (or re-record) rides with all four `Camera2`-`Camera5` active.
2. Train an `fxb`-conditioned model (e.g. `siam2d_3dhg_fxb`) on one baseline
   only - `--camera` picks one right camera per run, so a single training
   run never sees more than one baseline's disparities:
   ```bash
   python train.py --model siam2d_3dhg_fxb --data ~/datasets/airsim_stereo \
       --camera Camera2 --out runs/siam2d_3dhg_fxb_camera2
   ```
3. Evaluate that same checkpoint on one of the other, held-out baselines'
   (`Camera3`, `Camera4`, or `Camera5`) stereo pairs, each given its own true
   `fxb` -
   scored against the same `DepthPlanar` ground truth every camera in the
   rig shares, since all four right cameras and the ground-truth depth come
   from one `simGetImages` call (Section 2.3):
   ```bash
   python evaluate.py --ckpt runs/siam2d_3dhg_fxb_camera2/best.pth \
       --model siam2d_3dhg_fxb --data ~/datasets/airsim_stereo_test --camera Camera3
   ```
4. If `fxb`-conditioning works, accuracy on the held-out baseline should
   land close to the trained-on one. If the model only learned to expect
   whatever disparity range it was shown in training, held-out accuracy
   should collapse instead - the failure mode this whole conditioning
   scheme exists to prevent.

**Status: loader ready, test not yet run.** The `--camera` mechanism that
steps 2-4 need has been implemented and verified against a synthetic calib.json
matching `collect_dataset.py`'s real output format - correct baseline/folder
selection by both camera name and folder name, and a clear error on an
unknown one - but not yet exercised against a real recording or an actual
trained checkpoint. No AirSim data has actually been collected with this
settings file yet either (every completed sweep so far is TartanAir-only,
Section 15.3), so step 1 above is still the real blocker on running this
test for real.

---

### 3. SoA Paper

**Spotlight: Stereo Any Video** (Jing, Luo, Mao & Mikolajczyk, ICCV 2025)
[[jing2025stereoanyvideo]](#16-bibliography) is the closest published paper
to this project's goal, because it is the only recent method that directly
targets *stable-over-time* stereo depth from raw video, instead of solving
each frame separately - exactly what Section 1.3 says today's simple
averaging trick fails to do.

**What it does**: it extracts features with trainable encoders, plus a
**frozen monocular video-depth model** (Video Depth Anything) that supplies
a strong, already-stable sense of geometry without any stereo-specific
training. It builds a matching volume per frame and refines the disparity
step by step with a network (a "3D GRU") that looks across both the
matching volume and time together, then upsamples the result smoothly
across frames. The outcome is a disparity map that stays consistent from
frame to frame, instead of solving the scene from scratch 10 times a second
- the same complaint about frame-by-frame stereo that motivates
`TempoBandNet` in this project (Section 11.1), reached from a very
different, much larger-scale direction.

**Why it matters most for this problem**: it is the newest paper in this
line of work (2025), it directly targets frame-to-frame stability rather
than treating video as separate stereo pairs, and by leaning on a frozen
*monocular* model it partly avoids needing exact stereo calibration -
relevant to the calibration question in Section 1.5, even though it still
needs a calibrated stereo pair for its final matching step.

**Why it's a reference point, not the finish line, for this project**: it
runs on a large frozen encoder, the opposite of `FastStereoNet`'s
under-0.5M-parameter, edge-device target (Section 1.2, requirement 1); it
has no explicit drone motion input (it guesses motion from pixels alone,
unlike `TempoBandNet`'s use of the drone's own motion data); it does not
separately detect and fill blind spots (Section 1.3); and it has not been
tested on low-altitude drone footage at the distances (0.7-30 m) this
project cares about. **For plain single-frame accuracy** (the number every
stereo paper still reports on standard benchmarks), **IGEV-Stereo**
[[xu2023igev]](#16-bibliography) is still the one to beat - see Section 4.4
for both.

---

### 4. Related Work

#### 4.1 Pre-2020: Stereo Triangulation

| Method | Year | Strength | Weakness |
|---|---|---|---|
| Taxonomy & evaluation of dense two-frame stereo [[scharstein2002taxonomy]](#16-bibliography) | 2002 | Set out the four-step recipe (matching cost, combining costs, solving, clean-up) that people, including this project in Section 1.1, still use to describe stereo pipelines; created Middlebury, the first standard test set | Not a method itself - no learning involved, and no way to handle the hard regions it describes |
| Semi-Global Matching (SGM) [[hirschmuller2008sgm]](#16-bibliography) | 2008 | Copies global smoothness cheaply with several 1D passes in different directions - much better than simple block matching for a fraction of the cost; still runs in real time on a CPU today (OpenCV `StereoSGBM`, used optionally in this project's own pipeline) | Leaves streaking artifacts along the scan directions; still uses a hand-built matching rule, so it still fails on flat, repeating surfaces, exactly as described in Section 1.3 |
| MC-CNN [[zbontar2016mccnn]](#16-bibliography) | 2016 | First to learn how to compare two image patches with a small neural network, instead of hand-writing that rule, plugged into an otherwise classic pipeline; state of the art on KITTI/Middlebury at the time | Only replaces one step - still needs the same hand-built pipeline around it; comparing patches one at a time is slow |
| DispNetC / Scene Flow dataset [[mayer2016sceneflow]](#16-bibliography) | 2016 | First fully learned, single-pass network for disparity (a matching layer plus an encoder-decoder); also introduced the first synthetic dataset large enough to train it - the same "train on synthetic data, use it on real data" approach this project uses with AirSim | Its matching step is coarser than a full 3D matching volume, so it's noticeably behind GC-Net-style methods on fine detail |
| GC-Net [[kendall2017gcnet]](#16-bibliography) | 2017 | First to build an explicit 3D matching volume (disparity x height x width) from learned features and clean it up with 3D convolutions, then produce a smooth disparity value - the direct ancestor of `FastStereoNet`'s own `CostAggregation3D` (Section 7) | The full 3D matching volume uses a lot of memory and compute; far from real time when it was published |
| PSMNet [[chang2018psmnet]](#16-bibliography) | 2018 | Adds multi-scale context and a deeper 3D clean-up network, meaningfully improving accuracy in hard (occluded, texture-less) regions; became the standard baseline for years of later work | Even heavier than GC-Net; single-frame only, with no idea of time at all |

#### 4.2 Pre-2020: Monocular Depth Estimation

| Method | Year | Strength | Weakness |
|---|---|---|---|
| Make3D [[saxena2009make3d]](#16-bibliography) | 2009 | First to show a believable 3D reconstruction from a single image, using hand-built texture/gradient features and no stereo or motion at all | Assumes a fixed set of flat surface pieces; fails on scenes very different from what it was tuned on |
| Multi-scale deep network [[eigen2014depth]](#16-bibliography) | 2014 | First CNN approach: a rough global prediction refined by a second, more local one, trained with a loss that cares about relative depth more than exact scale | Blurry, low-resolution output; using a scale-agnostic loss is really an admission that a single camera alone can't know true scale |
| FCRN [[laina2016fcrn]](#16-bibliography) | 2016 | A fully learned, ResNet-based encoder-decoder with an efficient upsampling block - sharper, higher-resolution depth than the 2014 network above, with fewer parameters | Still needs dense ground-truth depth to train, which is expensive to collect on real data; still no scale guarantee outside its training data |
| Monodepth [[godard2017monodepth]](#16-bibliography) | 2017 | Removes the need for ground-truth depth entirely: trains on stereo pairs using how well the left image can be reconstructed from the right one - relevant here since it shows stereo geometry alone can supervise a network | Prone to errors at edges and occlusions (exactly the blind-spot problem in Section 1.3); depth is only as correctly scaled as the training camera rig, and it struggles on shiny/reflective surfaces |
| DORN [[fu2018dorn]](#16-bibliography) | 2018 | Turns depth prediction into picking one of several depth "buckets" instead of a raw number, with bucket sizes that grow with distance to match how depth gets harder to judge further away; state of the art on three benchmarks at once | Splitting depth into buckets introduces small errors at the bucket edges; still needs full supervision |
| Monodepth2 [[godard2019monodepth2]](#16-bibliography) | 2019 | Adds a smarter loss (robust to occlusion) and full-resolution training to Monodepth's self-supervised recipe; became the standard self-supervised baseline for years | Still only relative/up-to-scale depth, with no outside reference (see the calibration trade-off in Section 1.5); single-frame, with no consistency across a video |

#### 4.3 Datasets

| Dataset | Year | What it is | Why it matters here |
|---|---|---|---|
| KITTI 2012 [[geiger2012kitti]](#16-bibliography) / KITTI 2015 [[menze2015kitti15]](#16-bibliography) | 2012/2015 | Real driving data, sparse LiDAR ground truth | The standard "does it still work on new data" table every stereo paper reports (Section 15.2) |
| Middlebury 2014 [[scharstein2014middlebury]](#16-bibliography) | 2014 | Real indoor scenes, very precise ground truth | High-precision benchmark; balances out KITTI's outdoor/driving focus |
| Scene Flow [[mayer2016sceneflow]](#16-bibliography) | 2016 | Synthetic (computer-generated), about 35k frames | The original "training on synthetic data is enough" dataset - the direct precedent for training this project's models purely on AirSim data (Section 2) |
| ETH3D [[schops2017eth3d]](#16-bibliography) | 2017 | Real indoor and outdoor, laser-scanned ground truth | More scene variety beyond driving; another standard "new data" test |
| DrivingStereo [[yang2019drivingstereo]](#16-bibliography) | 2019 | Real driving data, about 180k frames | Hundreds of times bigger than KITTI stereo; useful if driving-specific tests are ever needed |
| NYU Depth V2 [[silberman2012nyuv2]](#16-bibliography) | 2012 | Real indoor colour + depth | Standard benchmark for single-camera depth (used by DORN, FCRN); relevant to Section 4.2, not to stereo |
| **TartanAir** [[wang2020tartanair]](#16-bibliography) | 2020 | Synthetic, made with AirSim, exact poses per frame | **The natural public benchmark for this project** - already supported by this repo's `stereo_datasets.py` (`TartanAirDataset`) and the #2 test tier in Section 15.2 |
| Mid-Air [[fonder2019midair]](#16-bibliography) | 2019 | Synthetic, low-altitude drone flight, 420k+ frames | Covers drone-specific altitudes/viewpoints that KITTI/Middlebury/ETH3D don't - close to this project's own flight range |
| UAVStereo [[zhang2023uavstereo]](#16-bibliography) | 2023 | Synthetic+real, drone, 34k+ pairs | The closest public drone-stereo dataset; used in Section 15.2 as a "does it generalize" test |

#### 4.4 Top Recent Methods (last 3 years)

**IGEV-Stereo** (Xu et al., CVPR 2023) [[xu2023igev]](#16-bibliography)
combines geometry, context, and fine matching detail into one compact
representation, then refines the disparity step by step (GRU-style) - it
was the #1 published method on KITTI 2012/2015 at the time, the fastest of
the top 10, and set a new best accuracy record on Scene Flow. It is the
accuracy benchmark this project's models should be compared against,
alongside the temporal methods below.

**TemporalStereo** (Zhang, Poggi, Tosi & Mattoccia, IROS 2023)
[[zhang2023temporalstereo]](#16-bibliography) is a coarse-to-fine network
that explicitly carries information from past frames forward - the same
"don't re-solve the scene from scratch every frame" idea `TempoBandNet` is
built around (Section 11.1), but reached with a learned warp instead of an
explicit motion-based shift.

**TC-Stereo** (Zeng, Yao, Wu & Jia, ECCV 2024)
[[zeng2024tcstereo]](#16-bibliography) points out that running stereo on
each video frame separately can give jumpy depth even when every frame
looks individually correct, and proposes a recurrent network (RAFT-style)
that carries state across frames to fix that, on top of just being
accurate.

**Stereo Any Video** (Jing et al., ICCV 2025)
[[jing2025stereoanyvideo]](#16-bibliography) - see Section 3 for the full write-up.

**LightStereo** (Guo, Zhang, Zhang, Zheng, Nie, Poggi & Chen, ICRA 2025)
[[guo2025lightstereo]](#16-bibliography) shows that a well-designed **2D**
matching network can match heavier 3D methods' accuracy on Scene Flow at
only 22 GFLOPs and 17 ms - the speed counterpart to IGEV-Stereo's accuracy
record, and the closest recent outside evidence for `FastStereoNet`'s own
2D/light-3D design (Section 7).

**ESMStereo** (Tahmasebi, Huq, Meehan & McAfee, 2025)
[[tahmasebi2025esmstereo]](#16-bibliography) targets the same real-time/
accurate trade-off as LightStereo, one year newer: its "Enhanced Shuffle
Mixer" disparity-upsampling block recovers detail a small-scale cost volume
would otherwise lose, without paying for a larger one - another data point
that the accuracy LightStereo/IGEV-Stereo get from a big cost volume can be
approximated much more cheaply, the same bet `FastStereoNet`'s 1/8-resolution
matching (Section 7) makes.

**PSKNet** (Liang, Hu, Hu, Xu & Chen, *Knowledge-Based Systems*, 2026)
[[liang2026psknet]](#16-bibliography) is the newest and most directly
relevant method found: real-time stereo depth measured **on a Jetson AGX
Orin**, the same edge-device class this project targets (Section 1.1,
requirement 1). Its per-pixel kernel-prediction refinement step
(`Kernel-Aware Residual Enhancer`) is aimed at the same problem
`FastStereoNet`'s edge-aware refinement stage is (Section 7): sharpen a
coarse, cheaply-matched disparity map back up using the input image. It has
no temporal component and does not address blind spots directly, so it
narrows the "fast/edge-ready" gap in Section 5's table without closing the
other columns.

**Fast-FoundationStereo** (Wen, Dewan & Birchfield, NVIDIA, 2025-2026)
[[wen2026fastfoundationstereo]](#16-bibliography) takes the opposite route
to speed: instead of designing a small network from scratch, it distills,
searches, and prunes down FoundationStereo (a large zero-shot foundation
stereo model) until it runs over 10x faster - 21 ms on an RTX 3090 with
TensorRT - while keeping most of the foundation model's zero-shot accuracy.
It's the speed-side mirror of Stereo Any Video's approach above (both start
from a large pretrained model rather than a hand-designed small one), and
another sign that "shrink a foundation model" is becoming a real alternative
to "design a small model directly" (this project's own approach, Sections
6-11) for hitting a real-time budget.

**Discussion.** These eight papers cover exactly the two things this project
cares about: IGEV-Stereo, LightStereo, ESMStereo, and Fast-FoundationStereo
mark different points on the accuracy/speed trade-off for *single-frame*
stereo (the last one via distilling a foundation model rather than designing
a small one from scratch), while TemporalStereo, TC-Stereo, and Stereo Any
Video show that frame-to-frame consistency is a real, active research
direction. PSKNet is the closest any of them come to this project's actual
deployment target, having been measured on the same Jetson-class hardware -
but it, like every other row here, has no temporal-consistency mechanism at
all. So the gap stands, just more precisely: nobody has combined edge-device
speed *and* frame-to-frame consistency in the same method, let alone added
an explicit blind-spot-filling output or a tested robustness study under
noisy motion data (Section 5 makes this exact).

#### 4.5 List of Method Papers

A plain list of every method paper mentioned above, with no extra
discussion - just the citation and a link. Datasets (Section 4.3) are not
methods, so they're not repeated here; full citation details for everything
below are in Section 16.

**Pre-2020 - Stereo Triangulation**
- Scharstein & Szeliski (2002). [A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms](https://link.springer.com/article/10.1023/A:1014573219977). *IJCV*.
- Hirschmüller (2008). [Stereo Processing by Semi-Global Matching and Mutual Information](https://ieeexplore.ieee.org/document/4359315) (SGM). *IEEE TPAMI*.
- Žbontar & LeCun (2016). [Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches](https://arxiv.org/abs/1510.05970) (MC-CNN). *JMLR*.
- Mayer et al. (2016). [A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation](https://arxiv.org/abs/1512.02134) (DispNetC). *CVPR*.
- Kendall et al. (2017). [End-to-End Learning of Geometry and Context for Deep Stereo Regression](https://arxiv.org/abs/1703.04309) (GC-Net). *ICCV*.
- Chang & Chen (2018). [Pyramid Stereo Matching Network](https://openaccess.thecvf.com/content_cvpr_2018/html/Chang_Pyramid_Stereo_Matching_CVPR_2018_paper.html) (PSMNet). *CVPR*.

**Pre-2020 - Monocular Depth Estimation**
- Saxena, Sun & Ng (2009). [Make3D: Learning 3D Scene Structure from a Single Still Image](https://www.cs.cornell.edu/~asaxena/reconstruction3d/saxena_make3d_learning3dstructure.pdf). *IEEE TPAMI*.
- Eigen, Puhrsch & Fergus (2014). [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network). *NeurIPS*.
- Laina et al. (2016). [Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/abs/1606.00373) (FCRN). *3DV*.
- Godard, Mac Aodha & Brostow (2017). [Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://openaccess.thecvf.com/content_cvpr_2017/html/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.html) (Monodepth). *CVPR*.
- Fu et al. (2018). [Deep Ordinal Regression Network for Monocular Depth Estimation](https://openaccess.thecvf.com/content_cvpr_2018/html/Fu_Deep_Ordinal_Regression_CVPR_2018_paper.html) (DORN). *CVPR*.
- Godard, Mac Aodha, Firman & Brostow (2019). [Digging Into Self-Supervised Monocular Depth Estimation](https://openaccess.thecvf.com/content_ICCV_2019/html/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.html) (Monodepth2). *ICCV*.

**Recent SoA (last 3 years)**
- Xu et al. (2023). [Iterative Geometry Encoding Volume for Stereo Matching](https://arxiv.org/abs/2303.06615) (IGEV-Stereo). *CVPR*.
- Zhang, Poggi, Tosi & Mattoccia (2023). [TemporalStereo: Efficient Spatial-Temporal Stereo Matching Network](https://arxiv.org/abs/2211.13755). *IROS*.
- Zeng, Yao, Wu & Jia (2024). [Temporally Consistent Stereo Matching](https://arxiv.org/abs/2407.11950) (TC-Stereo). *ECCV*.
- Jing, Luo, Mao & Mikolajczyk (2025). [Stereo Any Video: Temporally Consistent Stereo Matching](https://arxiv.org/abs/2503.05549). *ICCV*.
- Guo et al. (2025). [LightStereo: Channel Boost Is All You Need for Efficient 2D Cost Aggregation](https://arxiv.org/abs/2406.19833). *ICRA*.
- Tahmasebi, Huq, Meehan & McAfee (2025). [ESMStereo: Enhanced ShuffleMixer Disparity Upsampling for Real-Time and Accurate Stereo Matching](https://arxiv.org/abs/2506.21091). *arXiv preprint*.
- Liang, Hu, Hu, Xu & Chen (2026). [PSKNet: Lightweight Kernel-Aware Slice Network for Real-Time Stereo Depth Estimation on Edge Devices](https://www.sciencedirect.com/science/article/abs/pii/S0950705126000535). *Knowledge-Based Systems*, 336.
- Wen, Dewan & Birchfield (2025). [Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching](https://arxiv.org/abs/2512.11130). *arXiv preprint* (NVIDIA).

**Other referenced methods**
- Min et al. (2014). [Fast Global Image Smoothing Based on Weighted Least Squares](https://docs.opencv.org/4.x/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html). *IEEE TIP* - algorithmic basis of `STEREO_ENHANCE_WLS` (Section 1.1, Section 1.3).
- Yang et al. (2024). [Depth Anything V2](https://arxiv.org/abs/2406.09414). *arXiv preprint* - monocular foundation model referenced in the Section 5 gap table.

---

### 5. Gap in the Literature

No existing method - classic, learned single-frame, single-camera, or the
recent line of temporal-stereo papers - ticks every box from Section 1.2
at once:

| Method | Fast / edge-ready | Metric depth | Robust to noise | Post-proc baked in | Detects **and** fills blind spots | Temporal-aware | Low calibration burden |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| This repo's current SGM/BM + WLS pipeline (Section 1.1) | ✓ | ✓ | ✗ | ✗ | ✗ (NaN-masks only) | △ (motion-blind EMA) | ✗ |
| PSMNet / GC-Net [[chang2018psmnet, kendall2017gcnet]](#16-bibliography) | ✗ | ✓ | △ | ✓ | △ (implicit only) | ✗ | ✗ |
| IGEV-Stereo [[xu2023igev]](#16-bibliography) | △ | ✓ | ✓ | ✓ | △ (implicit only) | ✗ | ✗ |
| Monodepth2 / Depth Anything V2 [[godard2019monodepth2, yang2024depthanythingv2]](#16-bibliography) | ✓ | ✗ (up-to-scale) | ✓ | ✓ | n/a | ✗ / partial | ✓✓ (no stereo rig at all) |
| TemporalStereo / TC-Stereo [[zhang2023temporalstereo, zeng2024tcstereo]](#16-bibliography) | △ | ✓ | ✓ | ✓ | △ | ✓ | ✗ |
| Stereo Any Video [[jing2025stereoanyvideo]](#16-bibliography) | ✗ | ✓ | ✓ | ✓ | △ | ✓ | △ |
| LightStereo [[guo2025lightstereo]](#16-bibliography) | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| PSKNet [[liang2026psknet]](#16-bibliography) | ✓✓ (measured on Jetson AGX Orin) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| `TempoBandNet` (this repo, existing - Section 11) | ✓ | ✓ | ✓ | ✓ | ✗ | ✓✓ | △ |
| **Target method (this publication)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

*(✓ = yes, △ = partly, ✗ = no, ✓✓ = a specific strength of that method.)*

**Reading the table**: the classic-pipeline row shows exactly why Section
1.1's current approach needs replacing. The learned single-frame row
(PSMNet/GC-Net, IGEV-Stereo) is more accurate but still assumes one frame at
a time and still needs calibration. The single-camera row removes the need
for calibration but loses real-world scale entirely - not good enough for
obstacle avoidance (requirement 2). The temporal-stereo row (2023-2025) is
the newest and closest match on consistency, but none of those methods
target running on small edge hardware, and - checked against what each paper
actually claims in Section 4.4 - **none of them treat finding and filling
blind spots as a direct output**, only as something that might happen
incidentally as a side effect of better matching. PSKNet closes the
edge-hardware half of that gap on its own (Section 4.4) - it is the one row
here actually measured on Jetson-class hardware - but it is single-frame
only, with no temporal-awareness column filled in at all; no method found
fills both the "fast/edge-ready" and "temporal-aware" columns at once. This
project's own `TempoBandNet` already
covers the "fast" and "uses time" columns, but was never built to fill in
blind spots either (Section 11). Getting a ✓ in every column at once, backed
by a real test of robustness to noisy motion data and real latency numbers,
is the gap this project's models aim to fill - which is just another way of
saying the four things the original brief for this project asked for:
**robust, fills in blind spots with a real technique (not just a hack),
has all clean-up baked into the network, and gives accurate real-world
depth.**

This project's actual contribution is eight model architectures, all
implemented in this repository's [models/](models/) folder, that together
aim to close the gap in the table above (Section 6 `StereoConvNet`, Section
7 `FastStereoNet`, the `StereoConv3DNet` pair in Section 8-9, Section 10's
single-frame family, and Section 11 `TempoBandNet`). Four of those eight -
`FastStereoNet`, `MobileStereoNet`, `AnyStereoNet`, and `YoloStereoNet` -
also have an `fxb`-conditioned sibling variant: the same architecture,
additionally given the camera's focal-length x baseline as an input so one
trained model generalises across whatever stereo rig it's handed at
inference, instead of only the one baseline it was trained on (see
`FxbConditioning` in [models/common.py](models/common.py)). The other four
(`StereoConvNet`, both `StereoConv3DNet` window lengths, `TempoBandNet`)
don't have a separate sibling because they were already built `fxb`-
conditioned from the start - there was never a non-conditioned version to
compare against. Counting the four siblings separately, that is **twelve
`--model` choices** in total - the full one-row-per-choice breakdown, with
parameter counts, is Section 13's Model Summary table. All of them train
and evaluate the same way (Section 15):

| Model | Section | `--model` flag(s) | Idea | Params | Status |
|---|---|---|---|---|---|
| `StereoConvNet` | Section 6 | `siam2d_2dun_fxb` | DispNetC-style 2D correlation + U-Net cost aggregation | 3.79 M | implemented |
| `FastStereoNet` | Section 7 | `siam2d_3dhg` / `siam2d_3dhg_fxb` | Group-wise correlation + small 3D-conv hourglass + edge-aware refinement | 0.20 M | implemented |
| `StereoConv3DNet` | Section 8 | `c3d_3dhg_10_fxb` | C3D/I3D-style 3D convs over a 10-frame stacked volume | 0.15 M | implemented |
| `StereoConv3DNet` (T=3) | Section 9 | `c3d_3dhg_3_fxb` | Same network, a 3-frame stacked volume instead of 10 | 0.15 M | implemented |
| `MobileStereoNet` | Section 10.1 | `mobile2d_3dhg` / `mobile2d_3dhg_fxb` | Same pipeline, dense stages as MobileNetV2 inverted residuals | 0.13 M | implemented |
| `AnyStereoNet` | Section 10.2 | `pyr2d_casc2d` / `pyr2d_casc2d_fxb` | No 3D convs: coarse-to-fine residual disparity bands, anytime output | 0.14 M | implemented |
| `YoloStereoNet` | Section 10.3 | `yolo2d_3dhg` / `yolo2d_3dhg_fxb` | YOLO26 encoder/neck + multiplicative log-residual decoder | 2.61 M | implemented |
| `TempoBandNet` | Section 11 | `siam2d_egomotion_fxb` | Ego-motion-splatted prior + confidence-gated ConvGRU + narrow-band matching | 0.36 M | implemented |

*(where a row lists two flags, the `_fxb` one is the `fxb`-conditioned
sibling described above - same architecture and parameter count, one extra
input.)*

These eight architectures compare each other in five different ways.
`StereoConvNet` and `FastStereoNet` are the two single-frame reference
points, one slow, one fast: same family of ideas (2D matching + a
matching-cost network + a smooth disparity output), but `StereoConvNet`
stays mostly 2D and a few million parameters, while `FastStereoNet` moves
the clean-up step into a small learned 3D piece at a fraction of the size
(Section 7 explains the trade-off). `MobileStereoNet` and `AnyStereoNet`
each remove one of `FastStereoNet`'s two most expensive parts (the
full-resolution convolutions, and the 3D clean-up step), to see how cheap a
single-frame model can get. `YoloStereoNet` goes the other way, spending far
more parameters on features to see better at long range, where a 6 cm
camera baseline gives less than a pixel of disparity. `StereoConv3DNet`'s
two entries add a simple "just stack frames" temporal axis at two window
lengths (10 frames, 3 frames) - the same 0.15M weights either way, since
window length changes how much compute and memory it uses, not how many
parameters it has (Section 9). `TempoBandNet` adds the smarter,
motion-aware temporal axis those two are the simple comparison point for.
And, cutting across all of that, the four `fxb`-conditioned siblings add a
fifth axis - **does baseline-conditioning let one trained model stand in
for several rigs**, orthogonal to which architecture it's conditioning; the
TartanAir grid search (Section 15.3) sweeps every architecture with and
without it for exactly this reason. A full results table for this project
would report all twelve `--model` choices together - cheapest and simplest
to most capable, on every axis - all trained with the same objective,
chosen by the test in Section 15.3, so that model choice and loss choice
are never mixed up with each other.

---

### 6. Proposed Method - Slow Baseline Approach (StereoConvNet)

**`StereoConvNet`** ([models/stereo_conv_net.py](models/stereo_conv_net.py))
is a simple network built mostly from ordinary 2D convolutions. It's the
"obvious first thing you'd try" - a separate, slower and bigger model next
to this project's own `FastStereoNet` (Section 7). It isn't meant to be
turned into `FastStereoNet` later - both are trained and reported side by
side, as two different points of comparison.

**Inputs and outputs**: the left image, the right image, and the camera's
baseline distance `b`. The output is one dense depth map, in metres, for
that frame.

```
left  ──┐                                          baseline b (scalar)
        ├─ shared 2D-conv encoder (stride-2 x5) ──►  F_L, F_R  (1/4 res, 64 ch)
right ──┘                                                │
                                    1D correlation over disparity range
                                    (DispNetC-style: dot-product, no
                                     learned 3D conv here)
                                                │
                                 cost volume (D x H/4 x W/4), D disparity "channels"
                                                │
                         2D-conv encoder-decoder (U-Net, skip connections)
                         treats disparity as channels -> "mostly convolution
                         layers", no 3D conv, no recurrence, no attention
                                                │
                              coarse disparity @ 1/4 res
                                                │
                         learned upsample (transposed conv + skip connections)
                                                │
                              disparity map (full res, px)
                                                │
                    depth = focal(FOV, W) * b / disparity   [b is also
                    concatenated as a constant feature map at the bottleneck,
                    so the SAME trained weights are conditioned on whatever
                    baseline is given at inference time - see Section 1.5]
```

**Design notes**:

- The matching step is a simple comparison (a dot product at each candidate
  disparity, like DispNetC [[mayer2016sceneflow]](#16-bibliography)), not a
  learned 3D convolution. That keeps the network "mostly 2D convolutions",
  in clear contrast to Section 8's model, which uses 3D convolutions on
  purpose.
- Feeding `b` into the network (not just using it at the very end to turn
  disparity into depth) means one trained model can be tested across the
  different camera baselines this project records (Section 2.2) - this
  directly tests the calibration question from Section 1.5, instead of just
  assuming it works. In practice, the code feeds in `fxb` (focal length x
  baseline) rather than `b` alone, since every batch in this project already
  carries `fxb` as the one number disparity is computed from (Section 2.5) -
  so no new data needs to be added, and the model gets the same information
  either way.
- Loss: trained on disparity, not raw depth in metres. Which loss function
  is used is a `--loss` setting, not something built into the model itself
  (Section 14): the default, `smoothl1`, treats every pixel's error the
  same, while the log-space options exist because that is *not* the same as
  treating every pixel's depth error the same.
- Size: 3.79 million parameters (with `--max-disp 128`) - bigger than
  `FastStereoNet`'s 0.20 million (Section 7). This model isn't built to run
  on a small edge computer; it's here to be a clean, standard point of
  comparison. Making it smaller is future work, once it's clear how
  accurate it can get.

---

### 7. Proposed Method - Baseline Approach (FastStereoNet)

**`FastStereoNet`** ([models/baseline_net.py](models/baseline_net.py)) is
this project's small, single-pair model. It builds a matching volume with
"group-wise correlation" (explained below) and cleans it up with a small
learned 3D step - the fast twin to Section 6's `StereoConvNet`, built from
the same family of ideas but with about 1/20th the parameters. Its job in
this project is to isolate what `TempoBandNet`'s (Section 11) time-tracking
part actually adds, since the two models share every other piece. Unlike
`StereoConvNet`, it does not take the baseline distance as a direct network
input - only at the very last step, turning disparity into depth.

Two hard requirements drove the design: **small and fast** (real time on
the Jetson-class computer this project targets - under 0.5 million
parameters, works in half-precision, and doesn't scan the full disparity
range once `TempoBandNet` is running steadily) and **accurate** (able to
compete with much bigger stereo networks in the 0.7-30 m range that matters
for obstacle avoidance).

```
left ──┐
       ├─ shared CNN encoder ──► features at 1/8 resolution (32 ch)
right ─┘
            features L, R
                 │
   group-wise correlation cost volume        (8 groups x 16 disparity levels)
                 │
   small 3D-conv hourglass aggregation       (regularizes the volume)
                 │
   soft-argmin                               (differentiable winner-take-all)
                 │
   coarse disparity @ 1/8  ──► x8 bilinear upsample
                 │
   edge-aware refinement @ full res          (residual CNN conditioned on RGB)
                 │
   disparity map (full res, px)  ──►  depth = fx·B / disp
```

Design choices, in plain terms:

- **Matching at 1/8 resolution**: matching two images against each other is
  the most expensive part of stereo, and at 1/8 size it is 512x cheaper
  than at full resolution. The last step then sharpens the edges back up
  using the colour image (an idea from StereoNet).
- **Group-wise correlation** (from GwcNet): a richer way to compare left
  and right features than a plain dot product, without needing as many
  channels as simply gluing the features together.
- **Soft-argmin**: turns 16 rough matching scores per pixel into one smooth,
  sub-pixel disparity value that gradients can flow through during
  training - the same idea GC-Net [[kendall2017gcnet]](#16-bibliography)
  and PSMNet [[chang2018psmnet]](#16-bibliography) use (Section 4.1), just
  at a much smaller scale.

0.20 million parameters. Rough speed: about 90 ms on a single CPU thread at
448x256 - well over 100 FPS on a desktop GPU, and real time on Jetson in
half-precision. Compare that to Section 6's `StereoConvNet` at 3.79 million
parameters, and to LightStereo's published 17 ms / 22 GFLOPs (Section 4.4).

---

### 8. Proposed Method - Baseline Approach with Temporal Data (StereoConv3DNet)

**`StereoConv3DNet`** is the "using time" counterpart to Sections 6-7,
built mostly from **3D convolutions** over a stack of the last 10 frames.
Where Section 11's `TempoBandNet` is built to be *motion-aware* on purpose
(it shifts the last frame's disparity using the drone's known motion,
Section 11.2), this model is the plain, brute-force alternative: stack up
several raw frames and let 3D convolutions find whatever helps on their
own, with no motion input at all.

**Inputs and outputs**: the last 10 left frames and last 10 right frames
(about 1 second of history at this project's 10 FPS, Section 2.3), stacked
together, plus the baseline distance `b`. The output is one dense depth map
**for the most recent frame only** - the earlier frames are just extra
context to clean up that one estimate (directly addressing Section 1.3),
not a request for 10 separate depth maps.

```
left  frames [t-9..t] ──┐                              baseline b (scalar)
  (T=10, H, W, 3)        │
                         ├─ shared siamese 3D-conv encoder ──► F_L, F_R
right frames [t-9..t] ──┘    (Conv3D+BN+ReLU, 3x3x3 kernels,        (C, T', H/8, W/8)
  (T=10, H, W, 3)              stride-2 in space every 2 blocks,        T' << T
                               stride-2 in time twice: C3D/I3D-style)
                                          │
                    correlate F_L, F_R across candidate disparities
                    at the collapsed temporal features -> cost volume
                    (D x H/8 x W/8), already temporally denoised by
                    step above BEFORE any explicit matching happens
                                          │
                    3D-conv hourglass regularization (as GC-Net / PSMNet /
                    this repo's own CostAggregation3D) -> soft-argmin
                                          │
                        coarse disparity @ 1/8 res (current frame)
                                          │
              upsample + edge-aware refinement at full res, conditioned
              on the CURRENT frame's RGB only (same trick as StereoConvNet
              and FastStereoNet's refinement stage)
                                          │
                           disparity map (full res, px)
                                          │
                 depth = focal * b / disparity   [b conditioning as in Section 6]
```

**Design notes**:

- The 3D encoder is where requirement 5 from Section 1.1 ("use motion over
  time") actually happens: noise in a single frame gets averaged out by 3D
  convolutions across the 10-frame window *before* matching happens,
  instead of being smoothed *after the fact* the way the current
  `STEREO_TEMPORAL_SMOOTHING_ALPHA` averaging works today (Section 1.3).
- No motion input, no shifting of old data, no memory carried between
  calls - that's the point. This model answers a different question than
  `TempoBandNet` (Section 11): *does using several raw frames help at all*,
  versus *does tracking motion properly help even more, for less compute*.
  Both questions matter for this project's results.
- Cost: 3D convolutions cost roughly one extra dimension of compute over
  2D ones, on top of processing 10 frames instead of 1 - so this is
  expected to be the **most expensive to run** of the four models discussed
  in this project (`StereoConvNet`, `FastStereoNet`, `StereoConv3DNet`,
  `TempoBandNet`). That's actually a useful thing to show: it would prove
  that just throwing more raw frames at the problem costs the most *and*
  might still not beat the smarter, motion-aware approach for a fraction of
  the compute - exactly the case Section 11 makes for `TempoBandNet`.
- Loss: the same disparity loss as Section 6, chosen with `--loss`
  (Section 14), scored only against the ground-truth depth of the most
  recent frame in the 10-frame window.

---

### 9. Proposed Method - Baseline Fast Approach with Temporal Data (StereoConv3DNet, T=3)

**The same network as Section 8, just a shorter window.**
`c3d_3dhg_3_fxb` ([models/stereo_conv3d_net.py](models/stereo_conv3d_net.py))
has its own `--model` name, but it is *not* a second, different network -
it is `StereoConv3DNet` given 3 stacked frames instead of 10. The 3D
encoder already squashes down however many frames it's given into one
summary before matching happens (Section 8), so nothing about the model
changes - only how many frames the data loader stacks up before handing
them over. This isn't presented as a new design: it exists to make "how
many frames should we stack" a cheap, easy thing to test in the grid search
(Section 15.3), instead of a choice buried in a flag nobody sweeps over.

**Inputs and outputs**: the same as Section 8, but with 3 frames instead of
10 - the last 3 left frames and last 3 right frames (about 0.3 s of history
at this project's 10 FPS, with `--frame-stride 1`, Section 2.3), plus the
baseline `b`. The output is one dense depth map for the most recent frame.

**Why this comparison is useful.** Section 8 already expects
`StereoConv3DNet` to be the most expensive model in the project to run,
specifically because it runs 3D convolutions over every frame in its
window, and that cost grows with the window length - unlike `TempoBandNet`,
which reuses a small memory state instead of reprocessing old frames
(Section 11), or this model's own parameter count, which does *not* depend
on the window length at all (both `T=10` and `T=3` use the same 0.15
million weights; only compute and memory use grow with the window). `T=3`
is the cheapest window that still gives the encoder more than one
downsampling step to work with in time (Section 8's encoder halves the time
axis twice), so it's a natural floor: a much cheaper run that still checks
whether stacking raw frames helps at all, without always paying for the
full 10-frame window.

**Design notes**:

- Registered as `c3d_3dhg_3_fxb` in [models/__init__.py](models/__init__.py)
  and `MODEL_CHOICES` ([train_tartanair.py](train_tartanair.py)), so it is
  included by default in `--grid-models` alongside `c3d_3dhg_10_fxb`, and
  compared directly in the same sweep (Section 15.3).
- `--window` defaults to 3 for this model (`train.py`, `evaluate.py`,
  `train_tartanair.py`), the same way it already defaults to 10 for
  `c3d_3dhg_10_fxb` and 4 for `siam2d_egomotion_fxb` - no flag needed for the usual case.
- Since parameter count doesn't depend on window length, the fair way to
  compare it against `c3d_3dhg_10_fxb` is speed, GPU memory use, and accuracy
  - not size, since both are 0.15 million parameters.

---

### 10. Single-Frame Model Variants

The three models below each start from `FastStereoNet` (Section 7) and
change exactly one thing about it. They all take the same inputs and give
the same kind of output, so all four models (these three plus
`FastStereoNet` itself) can be swapped in with `--model` and compared
directly in a sweep (Section 15.3), and all four use the same disparity
convention and loss masking from Section 2.5. `MobileStereoNet` and
`AnyStereoNet` each remove one of `FastStereoNet`'s two most expensive
parts; `YoloStereoNet` instead spends *more* on features to see further.

| model | `--model` | params | what it changes |
|---|---|---|---|
| `MobileStereoNet` | `mobile2d_3dhg` | 0.13 M | dense stages become MobileNetV2 inverted residuals |
| `AnyStereoNet` | `pyr2d_casc2d` | 0.14 M | no 3D convs; coarse-to-fine residual disparity bands |
| `YoloStereoNet` | `yolo2d_3dhg` | 2.61 M | YOLO26 encoder/neck + log-residual decoder |
| `TempoBandNet` | `siam2d_egomotion_fxb` | 0.36 M | *not single-frame* - see Section 11 |

(Parameter counts at `--max-disp 128`, `--yolo-scale n`. `FastStereoNet`
itself is `siam2d_3dhg`, 0.20 M - see Section 7.)

#### 10.1 MobileStereoNet

`MobileStereoNet` ([models/mobilenet_stereo.py](models/mobilenet_stereo.py))
keeps `FastStereoNet`'s pipeline exactly the same - same matching step,
same 3D clean-up, same smooth-disparity step - and only rebuilds the two
parts that run over every single pixel: the feature extractor and the
full-resolution refinement step. Both are switched to MobileNetV2-style
lightweight blocks [[sandler2018mobilenetv2]](#16-bibliography) instead of
full-size convolutions. The small 3D clean-up step is left alone because it
is already small.

One detail is worth noting, because it's about memory use, not accuracy.
The feature-extractor blocks run at a small resolution, where using more
channels is cheap. The refinement step runs at full resolution, where the
same trick was using about 1.2 GB per block and going over 24 GB at batch
size 16 - so it uses a cheaper block there instead. That's a weaker
correction step, but it only nudges a disparity that's already been mostly
worked out, not the main estimate.

0.13 million parameters - the smallest of the four, and the natural choice
for the Jetson deployment target in Section 1.1.

#### 10.2 AnyStereoNet

`AnyStereoNet` ([models/anynet_stereo.py](models/anynet_stereo.py)) follows
AnyNet [[wang2019anytime]](#16-bibliography) and removes `FastStereoNet`'s
other expensive part: the 3D clean-up step. The idea is that the whole
disparity range only needs to be searched once, at the smallest, coarsest
scale, and everything after that can be a narrow, cheap search close to
that first guess.

```
stage 1 (1/16)  full disparity sweep (max_disp/16 levels), scored by *2D* convs
                over the volume flattened to (B, G*D, H, W), then soft-argmin
stage 2 (1/8)   warp right features by the upsampled stage-1 disparity, search a
                narrow band of residual offsets [-4,-2,-1,0,1,2,4] around it
stage 3 (1/4)   same again, narrower still [-2,-1,0,1,2]
refinement      the shared edge-aware full-resolution residual head
```

Flattening everything into plain channels is what lets ordinary 2D
convolutions do the clean-up work, and the narrow search bands only check a
handful of options instead of the whole range. Every stage produces its own
disparity map, so the network is *anytime*: stopping after stage 1 or 2
still gives a usable, rougher answer for a fraction of the cost - useful if
the flight controller has a deadline that changes with how busy it is. All
three stages are supervised together, using whichever loss Section 14
selects.

0.14 million parameters. Its feature extractor is deliberately smaller than
`FastStereoNet`'s: AnyNet spends its parameter budget on the multi-stage
search instead of on features.

#### 10.3 YoloStereoNet

`YoloStereoNet` ([models/yolo_stereo.py](models/yolo_stereo.py)) is the
newest model here, and the only one that spends *more* compute on features
instead of less. The reason is long range. On this project's actual camera
rig, disparity is 11.4 px at 2 m but drops to 2.3 px at 10 m and only 0.77
px at 30 m - past about 10 m, plain triangulation barely has any signal
left, so the model has to lean on a learned guess instead. That calls for a
stronger feature extractor than the simple one in
[models/common.py](models/common.py).

The stereo matching itself is unchanged - the same matching volume, 3D
clean-up, smooth-disparity step, and full-resolution refinement as
`FastStereoNet` (Section 7). What changes is the feature extractor and the
decoder, both taken from YOLO26's single-camera depth model
[[ultralytics2025depth]](#16-bibliography):

```
left  ─► YOLO26 backbone + PAN neck (layers 0..22)  ─► P3 (1/8) and neck P3/P4/P5
right ─► YOLO26 backbone, layers 0..4 ONLY          ─► P3 (1/8)
              │                                       (the right view is only ever
              │                                        used for matching, so it
              │                                        stops at P3)
   group-wise correlation ─► 3D hourglass ─► soft-argmin ─► disp @1/8
              │
   decoder: project neck P3/P4/P5 to a common width, fuse coarse-to-fine
   (bilinear x2 + add + 2 convs), inject disp/max_disp, ConvTranspose to 1/4
              │
   disp_4 = upsample(disp_8) * exp(residual)     ◄── log head, adapted
              │
   edge-aware refinement @ full res ─► disparity map (full res, px)
```

Two changes from the other models were made on purpose:

- **A multiplying correction instead of an adding one.** The other models
  (`FastStereoNet`, `MobileStereoNet`, `AnyStereoNet`) add a correction to
  the disparity. Adding treats every pixel's correction the same in raw
  disparity units, which is the same near-field bias the smooth-L1 loss has
  (Section 14). Multiplying instead treats the correction as a percentage
  of the current value, so it scales properly with distance. This is
  YOLO26-depth's own approach, adapted: where it outputs an absolute depth,
  here it outputs a percentage correction to a disparity that already has
  real-world scale from `fxb`.
- **The `max_disp` limit is relaxed.** The matching step still only
  searches up to `max_disp`, but the final output is no longer forced to
  stay under it - this matters because `--max-disp 128` would otherwise cut
  off anything closer than 0.63 m on TartanAir (Section 2.7).

⚠️ **A note on citations.** YOLO26's depth feature has no paper of its own,
and neither its docs nor its release notes cite earlier work. Its pieces
can still be traced: the SILog loss and its accuracy metrics come from
Eigen et al. [[eigen2014depth]](#16-bibliography), the multi-scale gradient
matching from MiDaS [[ranftl2020midas]](#16-bibliography), and training on
a mix of datasets with auto-labelled images from Depth Anything V2
[[yang2024depthanythingv2]](#16-bibliography) (which YOLO26-depth only uses
as a speed comparison). Any claim in this paper about that lineage should
cite those three sources, not just YOLO26's own docs.

The backbone and neck are read from
[models/yolo26_stereo.yaml](models/yolo26_stereo.yaml) - Ultralytics'
`yolo26-depth.yaml` with its depth-output line removed - using Ultralytics'
own model-building code, so the layer numbering matches the official
`yolo26*-depth` checkpoints exactly, and their pretrained encoder weights
can be loaded directly with no extra mapping needed. `--yolo-scale` picks
the model size (`n`: 2.61 M params, `s`: 9.39 M params).

Two things to keep in mind for any comparison table. First, at scale `n`
this has 2.61 million parameters against `FastStereoNet`'s 0.20 million, so
it's not a fair, like-for-like comparison and shouldn't be presented as
one. Measured speed at 480x640 on a GTX 1050 Ti is 53 ms against
`siam2d_3dhg`'s 48 ms - only 11% slower despite having 13x the parameters,
because both spend most of their time in the shared matching and
refinement steps, not the feature extractor. Second, it needs the
`ultralytics` package (which the other models do not), and `--crop` must be
a multiple of 32 instead of 16, since this backbone shrinks the image down
further.

---

### 11. TempoBandNet

`TempoBandNet` ([models/temporal_net.py](models/temporal_net.py)) is this
project's most advanced temporal model, and its main contribution: it uses
the drone's own motion to guess where things are, then only double-checks a
narrow range around that guess. This answers the "use motion over time"
requirement from Section 1.1 in a very different way than Section 8's
plain, brute-force `StereoConv3DNet` - shifting the last frame's data using
real motion and a learned "how much do I trust this" gate, instead of just
running 3D convolutions over stacked raw frames.

#### 11.1 Motivation

A stereo model that looks at each frame on its own re-solves the whole
scene from scratch 10 times a second, spending most of its effort
re-confirming things it already figured out 100 ms ago. A drone, though,
knows *exactly how it moved* between frames (from the simulator right now;
from onboard motion sensors on real hardware). Since most of the world
doesn't move, last frame's depth plus the drone's own motion is a strong
guess for this frame's depth. `TempoBandNet` is built entirely around using
that guess - carefully, not blindly.

#### 11.2 The temporal prior: forward splatting

Think of it like this: close your eyes, take one step forward, and guess
what the room looks like now. You'll be right almost everywhere - and just
as importantly, you'll know roughly *where* you can't be sure any more
(whatever that step just revealed).

Step by step, for each new frame:

1. Turn last frame's disparity map into a 3D point for every pixel (the
   scene, as the drone last saw it).
2. The world stayed still and the camera moved by a known amount - so move
   those points the same way the camera moved.
3. Project each point back onto the new image: every old pixel lands
   ("splats") on a new spot. Where several points land on the same pixel
   (something now blocks what used to be behind it), keep the closest one
   - the standard graphics trick called a **z-buffer**.
4. Pixels nothing landed on (the edge of the image sliding into view, or
   spots newly uncovered from behind an object) get marked as **not
   valid**. This mask is a real, geometry-based way to spot the "blind
   spot" regions Section 1.3 asks for - not just a side effect, but an
   actual detector for them.

The result is a per-pixel disparity **guess** (the "prior") plus a mask,
computed with no learned parameters at all. The memory from the previous
frame (a small learned state) is shifted the same way.

#### 11.3 Trust, but verify: the confidence gate

The guess is wrong wherever something moved, the motion data was noisy, or
last frame's estimate was already off. A small memory network combines the
shifted memory with the new frame's own evidence, and a learned **gate**
decides, pixel by pixel, how much to trust the guess:

```
init = gate · prior + (1 − gate) · coarse      (gate forced to 0 where mask = 0)
```

`coarse` is a full disparity search at 1/8 resolution, identical to
`FastStereoNet`'s (Section 7) - used to start from frame 0, and as a
per-pixel safety net at every frame after that. Training with fake motion
noise added on purpose (`--pose-noise`) teaches the gate to trust the guess
less when the motion data is unreliable - exactly the kind of robustness a
real onboard sensor needs, and this model's answer to the calibration
question in Section 1.5: it gets gradually less confident as motion data
gets worse, instead of assuming it's always perfect.

#### 11.4 Narrow-band matching

Once it has a trusted starting guess, the network does not search all 128
possible disparity values. It only checks **11 values spaced out around the
guess** (closer together near the guess, further apart away from it): tight
near the guess for fine detail, wider further out in case the guess drifted
a bit. Each of the 11 is scored the same way as `FastStereoNet`, and the
best one is picked the same smooth way too (Section 7) - just searching a
small, moving range instead of the whole fixed range. A second, narrower
5-value search at a higher resolution then sharpens the result, followed by
the same full-resolution edge clean-up `FastStereoNet` uses.

This is where the speed comes from: checking 11 + 5 values is much cheaper
than the full 16-level 3D search, and at run time the full search can be
skipped completely whenever the guess already covers the image (how often
that's safe to do, and what it costs in accuracy, is one of this project's
tests - Section 15.2).

#### 11.5 What flows between frames

The information carried from one frame to the next is just two things: the
small memory state, and the last disparity map. Gradients during training
only flow through the memory state, not through the shifted disparity map,
which keeps training stable across a multi-frame window.

0.36 million parameters - all of this project's temporal logic adds up to
just 0.16 million more than plain `FastStereoNet`. This result would fit
best at a robotics venue like ICRA or IROS, or a computer-vision workshop.

---

### 12. YOLO Techniques Used

No object detector runs anywhere in this project - "YOLO" here just names a
family of network designs, borrowed as a *feature extractor*, not as a
detector. Two different things travel under that name, from two different
places, and Section 10.3 already warns against mixing them up; this
section keeps them clearly separate.

**The original idea: detect everything in one pass.** The name comes from
Redmon et al.'s original YOLO paper
[[redmon2016yolo]](#16-bibliography) -
[arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640). It turned
object detection into a single regression problem: one pass of a CNN over
a fixed grid directly predicts all the boxes and confidence scores at once
- replacing the older two-step approach (first propose regions, then
classify each one separately). That traded a little accuracy for a big
speed win (45+ FPS on the hardware of the time), by removing that whole
second step. That speed-first trade-off - not any specific layer - is what
this project actually borrows: `YoloStereoNet` (Section 10.3) exists
because requirement 1 in Section 1.1 is real-time inference on Jetson-class
hardware, and the YOLO family is the best-proven existing answer to "how do
you make a detection-grade network fast," nine major versions later.

**What's actually reused: the YOLO26 backbone, neck, and size options.**
`YoloStereoNet` reads
[models/yolo26_stereo.yaml](models/yolo26_stereo.yaml) - Ultralytics'
`yolo26-depth.yaml` with its final depth-output line removed - using
Ultralytics' own model-building code, so the layers below are exactly what
YOLO26-depth ships, not a rebuild from scratch:

| stage | modules | role |
|---|---|---|
| stem | strided `Conv` blocks | shrinks the image twice before any heavier processing |
| backbone | `C3k2` blocks (split the input, run part of it through small bottleneck layers, then combine) at growing width | the main feature-extraction cost, at three resolutions (P3/P4/P5) |
| backbone tail | `SPPF` (pools at several scales) and `C2PSA` (a light self-attention block) | cheaply widens the model's field of view at the coarsest resolution (P5) |
| neck | upsampling + concatenation + `C3k2`, first top-down then bottom-up | mixes coarse, high-level information down into fine features, then fine detail back up into coarse ones, across all three resolutions |

`YoloStereoNet` reads the P3 features out of the backbone to build its
matching volume (the matching step itself is unchanged, Section 10.3) and
uses all three neck outputs (P3/P4/P5) as input to its decoder. The
**compound scaling** system - one config file plus a size setting per
letter (`n`/`s`/`m`/`l`/`x`) - is reused as-is through `--yolo-scale`,
instead of hand-writing a separate config for each size.

One thing this project does *not* claim: Ultralytics' YOLO26 release has no
paper of its own (Section 10.3), so `C3k2`/`SPPF`/`C2PSA` are described
here as the well-known general techniques they resemble - not credited to
one specific paper, the way `depth26`'s loss terms are credited in Section
14 (to `eigen2014depth` / `ranftl2020midas`). Any claim about where those
loss ideas came from belongs in Section 10.3 and Section 14, not here.

---

### 13. Model Summary

This section gives a one-stop, plain-language view of everything that gets
trained and tested in this project: every model (Section 13.1), and how
many combinations of settings actually get run in the grid search
(Section 13.2). See Section 7, Section 9 and Sections 10-11 for the full
detail behind each model.

#### 13.1 Models tested

All twelve models are already built and working (see the file column). None
of them are just on paper.

| `--model` | Class | Params | Uses several frames? | What it does, in plain words |
|---|---|---|---|---|
| `siam2d_2dun_fxb` | `StereoConvNet` | 3.79 M | No | Looks at one left/right pair. Uses only ordinary 2D building blocks - the slow, simple, easy-to-follow starting point (Section 6). |
| `siam2d_3dhg` | `FastStereoNet` | 0.20 M | No | Looks at one left/right pair. Uses a small 3D step to compare the two images. Small and fast, built to run on the drone's onboard computer (Section 7). |
| `siam2d_3dhg_fxb` | `FastStereoNetFxb` | 0.20 M | No | Same as `siam2d_3dhg`, but also told the camera's focal-length x baseline (`fxb`) as an input, so one trained model still works if it's later paired with a rig whose baseline distance is different from the training data's. |
| `c3d_3dhg_10_fxb` | `StereoConv3DNet` | 0.15 M | Yes - last 10 frames | Looks at the last 10 frames at once, with no memory carried between calls. The "just throw more frames at it" way of using time (Section 8). |
| `c3d_3dhg_3_fxb` | `StereoConv3DNet` | 0.15 M | Yes - last 3 frames | The exact same model as `c3d_3dhg_10_fxb`, just given 3 frames instead of 10 - cheaper to run (Section 9). |
| `mobile2d_3dhg` | `MobileStereoNet` | 0.13 M | No | Same idea as `siam2d_3dhg`, but built from smaller, cheaper building blocks (MobileNet-style). The smallest model in this project (Section 10.1). |
| `mobile2d_3dhg_fxb` | `MobileStereoNetFxb` | 0.13 M | No | Same as `mobile2d_3dhg`, plus the same `fxb` conditioning as `siam2d_3dhg_fxb` - the smallest model here that can still generalise across rig baselines. |
| `pyr2d_casc2d` | `AnyStereoNet` | 0.14 M | No | Same idea as `siam2d_3dhg`, but skips the 3D step and instead searches coarse-to-fine. Can give a quick, rough answer and improve it if there's time to spare (Section 10.2). |
| `pyr2d_casc2d_fxb` | `AnyStereoNetFxb` | 0.14 M | No | Same as `pyr2d_casc2d`, plus `fxb` conditioning. |
| `yolo2d_3dhg` | `YoloStereoNet` | 2.61 M | No | Same idea as `siam2d_3dhg`, but with a much bigger, stronger feature extractor (from YOLO26). Built to see further away, at the cost of being the biggest single-frame model here (Section 10.3). |
| `yolo2d_3dhg_fxb` | `YoloStereoNetFxb` | 2.62 M | No | Same as `yolo2d_3dhg`, plus `fxb` conditioning. |
| `siam2d_egomotion_fxb` | `TempoBandNet` | 0.36 M | Yes - remembers the past | Uses the drone's own motion to shift last frame's answer into place, checks a small area around it, and only trusts that guess as much as a learned "confidence" score says to. The most capable model in this project, and the winner of the first full test run (Section 11, Section 15.3). |

#### 13.2 Configuration tests: how much of the grid search actually runs

Training one model once is only one data point. This project instead runs
a **grid search**: it trains many combinations of model, batch size,
learning rate, and loss function, and compares the results. Not every
combination is worth trying, though - some would need more GPU memory than
the training machine has, so those are skipped automatically instead of
being run and left to crash.

| Setting | Choices | Count |
|---|---|---|
| Models | all 8 from the table above | 8 |
| Batch sizes | 8, 16 | 2 |
| Learning rates | 1e-3, 3e-3 | 2 |
| Loss functions (Section 14) | smoothl1, logl1, hybrid | 3 |
| **All combinations (8 x 2 x 2 x 3)** | | **96** |
| **Combinations expected to fit and actually run** | | **92** |
| **Combinations skipped (would run out of GPU memory)** | | **4** |

A third learning rate, `1e-2`, was tried too (Section 15.3) and dropped
from the default for the same reason `depth26` was: it was measured, not
just argued about. Across every combo tried, `1e-2` had the worst average
EPE and depth MAE of the three rates, and placed in zero of the top 10
results by depth MAE - it stays available as `--grid-lr ... 1e-2` for
anyone who wants to re-check it, but the default sweep no longer spends
compute on it.

All 6 skipped combinations are the same model at the same batch size:
`siam2d_egomotion_fxb` at batch size 16, with one of the two "log-space" losses
(`logl1`, `hybrid`). `siam2d_egomotion_fxb` is already the most memory-hungry model,
since it looks at several frames per training step, and those two losses
need a bit more memory on top of that - together that pushes past the
training machine's budget. `siam2d_egomotion_fxb` at batch size 16 with the plain
`smoothl1` loss still fits and does run.

A fourth loss, `depth26`, exists (Section 14) and can still be requested by
hand with `--grid-loss ... depth26`, but is no longer swept by default:
every combination tried so far gave noticeably worse EPE and depth MAE than
the other three losses, confirming what Section 14 already argued -
`depth26`'s scale-invariant SILog term is the wrong fit here, where
stereo's `fxb` already gives true metric scale. Including it would have
made the grid search 192 combinations (183 feasible) instead of 144, for
results not expected to place anywhere near the top.

"Expected to fit" is a prediction, not a guarantee for every model: it's
based on GPU memory actually measured for `siam2d_3dhg`, `pyr2d_casc2d`,
`mobile2d_3dhg`, `yolo2d_3dhg`, and `siam2d_egomotion_fxb`. The three newer models
(`siam2d_2dun_fxb`, `c3d_3dhg_10_fxb`, `c3d_3dhg_3_fxb`) have not been measured
yet, so they are assumed to fit for now, and the training script will
report honestly (as `FAILED`, not a crash of the whole run) if one of them
turns out not to.

---

### 14. Loss Functions

A loss function is just the number training tries to shrink: it scores how
wrong today's prediction is, and every weight update is a step that would
have made that number a little smaller. This project trains on
**disparity** rather than raw depth (Section 2.5), and lets `--loss`
(`losses.py`) pick which of four scoring rules does that - all four still
score every model's finest and coarser (`aux`) disparity maps together, so
switching `--loss` never changes *what* is supervised, only *how* the error
is measured.

**Why the choice matters here, not just in theory.** Depth is not a linear
function of disparity - `depth = fxb / disp` - so a scoring rule that treats
every pixel of disparity error the same treats a nearby obstacle and a
distant one very differently in the metric that actually matters (metres).
On this project's own rig, a constant **1 px** disparity error is a few
centimetres of depth error up close but tens of metres of depth error far
away:

![A constant 1-pixel disparity error grows into a huge depth error at range, while a constant percentage error stays proportional at any range](figures/depth_error_vs_range.png)

The left panel is what training with plain smooth-L1-on-disparity is really
pushing for: the same pixel accuracy everywhere, which is a poor stand-in
for the same *depth* accuracy everywhere, since a 1 px error means almost
nothing at 2 m but tens of metres at 90 m. The right panel is what the
log-space options push for instead - a fixed *percentage* error, which
grows in a sensible, predictable way with range instead of exploding.
Three of the four `--loss` options exist specifically to get from the left
picture to the right one.

#### `smoothl1` - the default

Smooth-L1 (also called Huber loss) behaves like a squared error (L2) for
small mistakes and like an absolute error (L1) for large ones, switching
over at a threshold (`beta`, here 1 px):

![L1 loss grows linearly with error, L2 grows quadratically, and smooth-L1 follows L2 near zero and switches to L1's shallower slope past beta=1](figures/loss_l1_l2_smoothl1.png)

- **L2** alone gives a very informative, smoothly-varying gradient near zero
  (good for fine convergence), but a single huge outlier - an occluded
  pixel, a sky pixel near the far clip, a bad ground-truth value - squares
  its error and can dominate an entire batch's gradient.
- **L1** alone fixes that (constant gradient magnitude regardless of how
  wrong a pixel is), but its gradient never shrinks near zero, so it can
  make final convergence noisier.
- **Smooth-L1** gets both: L2's stable, informative gradient for the
  pixels that are already close, and L1's bounded gradient for the rare
  pixel that is very wrong.

**Why it seems fitting here**: this project's ground truth comes from a
renderer (Section 2.2) and, later, a real stereo rig - both have occasional
bad or occluded pixels, and a model in early training produces plenty of
large, uninformative errors of its own. Smooth-L1 keeps those from blowing
up the gradient while still training precisely once predictions are close.
It is also this project's original, simplest loss, kept as the default so
earlier runs and checkpoints stay comparable. Its blind spot is exactly the
left panel above: it is silent about *where* in the depth range that pixel
error lands.

#### `depth26` - the faithful reference port

`depth26` ports YOLO26's monocular `DepthLoss26`
[[ultralytics2025depth]](#16-bibliography) onto log-disparity: instead of
scoring the raw pixel difference, it scores the difference in
**log(disparity)**, plus a term that compares the *slope* of the predicted
and true disparity maps (SILog
[[eigen2014depth]](#16-bibliography) + multi-scale gradient matching
[[ranftl2020midas]](#16-bibliography); [losses.py](losses.py) has the exact formula).
Because `log depth = log(fxb) - log(disp)` for stereo, an error in
log-disparity *is* the same error in log-depth - so this is precisely the
"right panel" behaviour above, for free.

**Why it seems fitting here, with a catch**: the relative-error idea is
exactly what Section 2.5's non-linearity calls for. The catch is SILog's
extra step: it also subtracts the *mean* error across the image before
scoring, which makes the loss **scale-invariant** - the correct choice for
monocular depth, where a network has no way to know true metric scale at
all, but the wrong choice here, where stereo calibration (`fxb`) already
hands the network true metric scale for free. Throwing that away is a step
backward for this problem, not an improvement, and this is no longer just
an argument from the formula: every combination tried in this project's
grid search (Section 15.3) came back with EPE and depth MAE clearly worse
than the other three losses (EPE 6.8-47.8px against ~3.0-3.2px, depth MAE
2.7-6.2m against ~1.5-1.7m). Because of that, `depth26` was dropped from
the grid search's default loss list - it stays available as `--loss
depth26` or `--grid-loss ... depth26` for anyone who wants to check it
again, but the default sweep no longer spends compute on it.

**A closer look: what "scale-invariant" actually throws away.** In plain
terms, SILog looks at the log-error on every pixel, `d = log(pred) -
log(ground truth)`, then subtracts the *average* of `d` across the whole
image before scoring (`losses.py`'s `_silog`, with the default
`--loss-silog-lambda 1.0`). Subtracting that average is the entire trick:
if a model's disparity is uniformly too high or too low by some constant
factor - the same wrong scale on every pixel - that constant cancels out
in the subtraction and the loss barely notices. For a single camera, this
is the right call: it has no way to know true scale in the first place, so
a loss that ignores a uniform scale error isn't giving anything up. For
stereo, that constant is exactly `fxb` - it's known from calibration, not
guessed, and it's precisely the number `depth = fxb / disp` needs to be
right. A stereo network *can* get that constant right, because the
geometry hands it over for free (Section 2.5). By subtracting it out,
SILog tells the network not to bother getting it right - which is what
"throwing away `fxb`" means concretely, and it lines up with what was
measured: `depth26`'s EPE (which is sensitive to a whole-image scale
drift) was far worse than its depth MAE relative to the other losses,
exactly the pattern a model would show if it settled on the right *shape*
of the disparity map but the wrong overall scale. Setting
`--loss-silog-lambda 0.0` removes this effect entirely (SILog becomes
plain log-RMSE, which does score the mean offset) - a cheaper way to test
this specific claim than switching to `logl1` outright, though `logl1`
also drops the multi-scale-only SILog framing in favour of importing
plain L1 error too, so it isn't a controlled swap of only this one term.

#### `logl1` - the same idea, without the mismatch

`logl1` keeps `depth26`'s log-disparity framing and gradient-matching term,
but replaces SILog with plain **L1 in log-disparity space** - no
mean-subtraction, so it stays scale-*aware*.

**Why it seems fitting here**: this is the "right panel" behaviour above
*and* keeps the absolute metric scale `fxb` already supplies, which is
exactly the combination this project's own calibrated rig can use but
monocular methods cannot. It targets the metrics that actually matter for
obstacle avoidance (abs-rel depth error, depth MAE) far more directly than
scoring raw disparity pixels ever can.

#### `hybrid` - both, weighted

`hybrid` simply adds smooth-L1 on disparity to a weighted `logl1` term (plus
the gradient term): `smoothl1 + w_log * logl1 + w_grad * gradient`
(`--loss-w-log`, `--loss-w-grad` control the weights).

**Why it seems fitting here**: the near field is where an obstacle is
already close enough that raw pixel precision matters most and disparity is
large (so smooth-L1's bias there is small anyway, per the left panel); the
far field is where the relative-error framing matters most. `hybrid` does
not have to pick one regime over the other - it gets smooth-L1's
near-field precision and stability for free while adding the log term's
far-field relative accuracy on top, with `--loss-w-log` as a single knob to
tune the balance between them instead of committing to one loss for the
whole depth range.

#### Why every option beyond `smoothl1` also scores gradients

None of the three pointwise scores above - smooth-L1, SILog, plain L1 -
reward getting *edges* right: a prediction can have a low average pixel
error while still smearing a thin obstacle (a branch, a wire, a pole) into
the background behind it, because the handful of pixels on that boundary
barely move the average. The gradient-matching term instead compares the
*difference between neighbouring pixels* in the predicted and true
disparity maps - i.e. it scores whether edges land in the same place - at
several progressively coarser resolutions, so it catches both a sharp
nearby edge and a large-scale structural boundary a single resolution would
miss.

**Why it seems fitting here**: a thin obstacle disappearing into the
background is precisely the dangerous failure mode this whole project
exists to avoid (Section 1.1, requirement 4) - a pointwise loss has no way
to penalise it directly, so `depth26`, `logl1` and `hybrid` all carry this
term specifically to reward disparity maps that keep obstacle edges sharp
rather than merely being accurate on average.

| `--loss` | penalises | fits because |
|---|---|---|
| `smoothl1` | raw disparity, robust to outliers | stable against occlusions/bad GT; the simple, comparable default |
| `depth26` | relative (log) disparity, but scale-invariant | throws away the metric scale `fxb` already gives us - measurably worse in this project's own tests, so off by default (Section 15.3) |
| `logl1` | relative (log) disparity, scale-aware | relative error *and* keeps `fxb`'s metric scale - the combination this project can actually use |
| `hybrid` | both, weighted | near-field precision (smoothl1) and far-field relative accuracy (logl1) together, tunable |

Metrics are unaffected by the choice, so runs trained with different losses
remain directly comparable on val EPE, D1 and depth MAE - only `train_loss`
itself is not comparable across them.

---

### 15. Training and Evaluation

Every model (Section 6 through Section 11) uses the same disparity ground
truth and loss-masking rules from Section 2.5, and the same
`forward -> {"disp", "aux"}` interface, so all of them train in one shared
loop and can be compared directly. This section covers how they are
trained (15.1), tested (15.2), swept across settings (15.3), and run from
the command line (15.4).

#### 15.1 Training

| aspect | setting |
|---|---|
| loss | chosen with `--loss` (default `smoothl1`, the original); see Section 14 |
| what gets supervised | the final disparity map (full weight) plus each rougher in-between map (partial weight), all compared at full resolution |
| optimizer | AdamW, learning rate 4e-4, weight decay 1e-4, cosine decay, gradient clipping at 1.0 |
| batch size | 8 (single-frame models) / 4 windows (temporal models), mixed precision optional |
| data augmentation | consistent brightness/contrast/colour changes across a window; random crops (multiples of 16 pixels - **32** for `YoloStereoNet`, since its backbone shrinks the image further) |
| validation | held out **by ride** (Section 2.4); best model picked by validation EPE (average disparity error) |

**Training objective** (`losses.py`): chosen with `--loss`, default
`smoothl1`. What each of the four options scores, and why it fits this
project's non-linear depth/disparity relationship, is explained in full in
Section 14. The metrics aren't affected by which loss is used, so runs
trained with different losses stay directly comparable on validation EPE,
D1, and depth MAE - only the raw training loss number itself isn't
comparable across them.

**Temporal specifics** (`StereoConv3DNet`, Sections 8-9; `TempoBandNet`,
Section 11) - `TempoBandNet` trains on a **window of 4 frames**
(`--frame-stride 2` gives 0.2 s between frames, so there's real motion to
learn from), run one frame at a time through its memory; the loss is
averaged over all 4 frames, so frame 0 trains the cold-start path and
frames 1-3 train the memory/gate/search path, in every single training
example. Gradients only flow between frames through the memory state, not
through the shifted disparity map. `StereoConv3DNet` instead takes its
whole window in one go, 10 frames (`c3d_3dhg_10_fxb`, Section 8) or 3 frames
(`c3d_3dhg_3_fxb`, Section 9), with no memory carried between calls;
`--window` defaults accordingly, and `--frame-stride 1` gives the
"about 1 s" / "about 0.3 s" of history each section describes (this
project's usual default of 2 is tuned for `TempoBandNet`, not this pair).
**Pose-noise training** adds small random rotation (about 0.3°) and
proportional position noise to the motion data fed to `TempoBandNet`,
during training only.

**Planned order of experiments** (each step is also a row in the results table):

0. First, settle on one loss (Section 15.3): test all four `--loss` options
   on the established models, so every result after this uses one chosen
   loss and model choice and loss choice never get mixed up together;
1. `StereoConvNet` (Section 6) and `FastStereoNet` (Section 7) on AirSim data -
   the two single-frame reference points;
2. `MobileStereoNet` and `AnyStereoNet` (Sections 10.1-10.2) - how cheap can a
   single-frame model get before accuracy breaks down?
3. `YoloStereoNet` (Section 10.3) - does a much stronger feature extractor
   actually help in the 10-30 m range, where a 6 cm camera baseline gives
   less than a pixel of disparity? Compare against a model of similar size,
   not just against `siam2d_3dhg`;
4. `TempoBandNet` (Section 11) on AirSim, with no added motion noise - the
   best case, with perfect motion data;
5. `TempoBandNet` with motion noise added - a robustness curve;
6. `c3d_3dhg_10_fxb` vs `c3d_3dhg_3_fxb` (Sections 8-9), 10 frames vs 3
   frames, same parameter count either way - does stacking raw frames help
   at all, and how much of its extra cost is actually worth it, compared to
   `TempoBandNet`'s cheaper, motion-aware approach;
7. all eight models trained on TartanAir (Section 2.7), and on TartanAir
   plus this project's own data - does the project's own low-altitude
   AirSim data (Section 2) actually help?

Run with:

```bash
python train.py --model siam2d_3dhg --data ~/datasets/airsim_stereo --out runs/siam2d_3dhg
python train.py --model siam2d_2dun_fxb --data ~/datasets/airsim_stereo --out runs/siam2d_2dun_fxb
python train.py --model c3d_3dhg_10_fxb --frame-stride 1 \
    --data ~/datasets/airsim_stereo --out runs/c3d_3dhg_10_fxb
python train.py --model c3d_3dhg_3_fxb --frame-stride 1 \
    --data ~/datasets/airsim_stereo --out runs/c3d_3dhg_3_fxb
python train.py --model siam2d_egomotion_fxb --window 4 --bs 4 --pose-noise 0.3 \
    --data ~/datasets/airsim_stereo --out runs/siam2d_egomotion_fxb

# a non-default objective, and the YOLO26-encoder model
python train.py --model pyr2d_casc2d --loss logl1 \
    --data ~/datasets/airsim_stereo --out runs/anynet_logl1
python train.py --model yolo2d_3dhg --loss hybrid --crop 480x640 \
    --data ~/datasets/airsim_stereo --out runs/yolo_hybrid
```

(`--dataset tartanair --data <root>` switches to TartanAir directly, though
[train_tartanair.py](train_tartanair.py) is the fuller entry point for that
corpus - see Section 2.7. Crops must be multiples of 16, or of 32 for
`--model yolo2d_3dhg`; native 256x448 satisfies both.)

#### 15.2 Evaluation

**Metrics**: EPE (average disparity error, in pixels), D1 (percent of
pixels that are off by more than 3 px and more than 5%), and depth error as
a percentage, split by range (0-10 m, 10-30 m, 30-95 m). Splitting by range
is more honest about obstacle avoidance than a single disparity number,
since the same disparity error turns into a much bigger depth error far
away than it does close up.

**Tests, in order of importance**:

1. held-out AirSim rides (environments not seen during training);
2. **TartanAir** [[wang2020tartanair]](#16-bibliography) - a public dataset,
   also made with AirSim, with known poses per frame, so `TempoBandNet` can
   be tested on it end to end and numbers stay comparable
   (`TartanAirDataset` is already built into `stereo_datasets.py`);
3. does it still work on data it's never seen anything like: the standard
   KITTI 2015 / Middlebury / ETH3D test sets (Section 4.3), plus the
   drone-specific UAVStereo and Mid-Air datasets
   [[zhang2023uavstereo, fonder2019midair]](#16-bibliography);
4. **speed**: half-precision on Jetson (`evaluate.py --bench`), reported
   per model, and for `TempoBandNet`, both a cold start and steady running
   speed;
5. **cross-baseline generalization** for the four `fxb`-conditioned models:
   trained on one camera baseline, evaluated on another it never saw
   (Section 2.8) - proposed, not yet run.

**Tests that change one piece of `TempoBandNet` at a time**: remove the
motion-based guess entirely (leaving just `FastStereoNet` plus memory),
remove the trust gate (always trust the guess), change how many disparity
values are checked around the guess (5, 11, or 21), change the frame gap
at test time (does the guess still work with a bigger 0.5 s gap?), and add
noise to the motion data at test time.

```bash
python evaluate.py --ckpt runs/siam2d_egomotion_fxb/best.pth --model siam2d_egomotion_fxb \
    --data ~/datasets/airsim_stereo_test --window 8
python evaluate.py --model siam2d_egomotion_fxb --bench --size 256x448 --fp16   # latency
```

#### 15.3 Grid search

`train_tartanair.py --grid-search` tries every combination of
**`--grid-models` x `--grid-bs` x `--grid-lr` x `--grid-loss`**, replacing a
shell script this project used before. `--model/--bs/--lr/--loss` are
ignored in this mode, and `--out` becomes the folder the whole sweep is
saved under.

**Each combination runs as its own separate process**, not inside the main
one. That's the whole reason this exists: if one combination runs out of
GPU memory or crashes, it can't take down the ones after it - a sweep that
hits a bad combination still finishes every other one instead of dying
partway through. A failed combination is just marked `FAILED`, and the
sweep keeps going.

**A sweep can be paused and picked back up later.** Each run writes
`log.csv` after every epoch. On a re-run, a combination whose `log.csv`
already has enough rows is marked `SKIPPED`, and a combination that got
partway through resumes from its last saved checkpoint instead of starting
over. So an interrupted sweep can simply be started again.
`--grid-force-rerun` forces everything to redo anyway.

**Fair comparison between combinations** comes from `--max-train-windows`
and `--max-val-windows`: they cut the data down to a fixed number of
examples, using the same fixed random choice every time, so every
combination trains and checks itself on the exact same data. Checking
happens every `--eval-every` epochs, since on a big dataset that step can
take longer than training itself, so it's controlled separately.
`--grid-epochs` (how long each sweep cell trains) is kept separate from
`--epochs` (used for the one full, final run after the sweep has picked a
winner).

**Output.** A `summary.csv` file at the top of the sweep folder, updated
after every combination finishes (so even an interrupted sweep leaves a
readable record), with one row per combination:

```
model,batch_size,lr,loss,status,best_val_epe,final_val_epe,best_val_depth_mae,seconds
```

Depth MAE is tracked alongside EPE because the two numbers can disagree: a
loss that ignores absolute scale (`--loss depth26`, Section 14) can look
genuinely good while its raw disparity number (and so its EPE) drifts off -
sorting only by EPE would unfairly throw it out. A sorted summary is
printed at the end, and also if you stop it early with Ctrl-C. Each run's
folder is named `{model}_bs{bs}_lr{lr}`, plus the loss name if it isn't the
default; TensorBoard (a training dashboard) starts automatically for the
whole sweep (`--no-tensorboard` turns it off) and is reachable from other
machines on the same network.

Measured peak VRAM at 480x640 with `--amp`, for choosing `--grid-bs`:

| model | bs 8 | bs 16 | bs 32 |
|---|---|---|---|
| `siam2d_3dhg`, `pyr2d_casc2d` | 2.7 GB | 5.5 GB | 11 GB |
| `mobile2d_3dhg` | 4.8 GB | 9.5 GB | 19 GB |
| `yolo2d_3dhg` | 3.1 GB | ~6 GB* | ~12 GB* |
| `siam2d_egomotion_fxb` | 11 GB | 22.5 GB | OOM |

`siam2d_egomotion_fxb` looks at 4 frames per training step, so it uses about 4x the GPU
memory of a single-frame model. *`yolo2d_3dhg` was only measured at batch size 8
(the GPU used for measuring only has 4 GB); the other two numbers are a
straight-line guess from the pattern the other rows follow, and should be
re-measured on the real training machine. `siam2d_2dun_fxb`, `c3d_3dhg_10_fxb` and
`c3d_3dhg_3_fxb` aren't in this table yet - none of them have a measured
number, so the sweep lets any batch size through for them and will report
honestly if one runs out of memory, the same as it would for any other
unmeasured model. `c3d_3dhg_10_fxb` is the most likely of the three to need
its own row once measured: like `siam2d_egomotion_fxb`, its window length (10 frames,
vs. `c3d_3dhg_3_fxb`'s 3) directly multiplies how much memory its 3D
convolutions use.

The recommended order is to **settle on one loss before comparing
models**, so a later result about which model is best can't accidentally
be caused by which loss it happened to use:

```bash
# 1. which objective? four losses x the four established models
python train_tartanair.py --grid-search --data ~/datasets/tartanair \
    --grid-models siam2d_3dhg siam2d_egomotion_fxb mobile2d_3dhg pyr2d_casc2d \
    --grid-loss smoothl1 depth26 logl1 hybrid \
    --grid-bs 8 --grid-lr 3e-4 --grid-epochs 10 --crop 480x640 --amp \
    --out runs/loss_ablation

# 2. then the architecture sweep, at the winning objective
python train_tartanair.py --grid-search --data ~/datasets/tartanair \
    --grid-models siam2d_3dhg siam2d_egomotion_fxb mobile2d_3dhg pyr2d_casc2d yolo2d_3dhg \
    --grid-epochs 12 --crop 480x640 --workers 1 --amp \
    --max-train-windows 1000 --max-val-windows 1000 \
    --out runs/tartanair_grid
```

**First finished sweep** (`runs/tartanair_grid`, 4 models x batch size
{8,16} x learning rate {1e-3,3e-3} x 3 losses, 44 of 48 combinations run -
4 skipped for using too much GPU memory): the winner was `siam2d_egomotion_fxb`, batch
size 8, learning rate 3e-3, loss `logl1` (depth MAE 1.39 m). The choice of
loss mattered far more than the choice of model - every one of the top 26
results used `logl1` or `hybrid`, and the best `smoothl1` result only
ranked 27th. So which model wins matters much less than getting off the
plain disparity loss.

Two changes were made to the defaults based on that result, instead of
leaving them as one-off flags: **`yolo2d_3dhg`** (Section 10.3) is now included by
default in `--grid-models` - the first sweep only compared models with no
temporal or long-range-specialized feature extractor - and **`depth26`**
(Section 14) was made a default `--grid-loss` entry too, so whether it
really is a bad fit for stereo would get measured on this project's own
data, not just argued from theory.

**Second sweep** (`runs/tartanair_grid`, 8 models x {8,16} x
{1e-3,3e-3,1e-2} x 4 losses): `depth26` was measured across every model
tried so far and came back clearly worse every time (EPE 6.8-47.8px and
depth MAE 2.7-6.2m, against ~3.0-3.2px and ~1.5-1.7m for the other three
losses) - confirming the theoretical argument above, not just repeating
it. Based on that, `depth26` was removed from `--grid-loss`'s default
again (it stays available as an explicit choice), and the sweep was
restarted to pick up the new default; the completed `depth26` runs
themselves were kept, not deleted, since they're the evidence for this
change.

`1e-2` (added after the first sweep because the winning rate, 3e-3, sat at
the edge of the range tested, so the true optimum was never bracketed) got
the same treatment once it had actually been tried: across every combo
run, it had the worst average EPE and depth MAE of the three rates and
placed in zero of the top 10 results by depth MAE. So it was dropped from
`--grid-lr`'s default too, and the sweep restarted a second time - the
range was extended to check whether 3e-3 was the peak, and the answer
turned out to be yes.

**Round 1 conclusions: what 197 evaluated configs actually show.** The
narrative above is the chronological "what was run and why." This is the
analysis of the result: every run under `runs/tartanair_grid` re-evaluated by
[eval_all.py](eval_all.py) - held-out-environment accuracy, inference time,
and an overfit/underfit read of each run's own training curve, one row per
(model, batch size, lr, loss) in `eval_all.csv` - plus the raw per-epoch
numbers in each run's `log.csv`. 350 GPU-hours went into the 187 combos that
actually trained to completion. A caveat before any of it: this entire round
used the sweep's cheap proxy settings (`--data-fraction 0.3`, 20 epochs,
Section 15.3's own default), not a full training run - what follows ranks
configs *relative to each other under that proxy*, which is a different
claim from ranking their best achievable accuracy.

*Safe, robust results:*

- **Loss**: `logl1` beats `hybrid` in 61 of 66 head-to-head comparisons at
  matching model/batch-size/lr (median +0.13 m depth MAE, mean +0.23 m) -
  the single most consistent result in the whole round. Both beat
  `smoothl1` (n=41, mean 5.61 m vs `logl1`'s 1.55 m) and `depth26` (n=5,
  mean 3.93 m) by a wide margin, confirming the earlier sweep findings
  above at full scale rather than on a handful of models.
- **Architecture**, restricted to `logl1`/`hybrid` only so `depth26` and
  `smoothl1`'s outliers can't skew it: `siam2d_2dun_fxb` (`StereoConvNet`,
  Section 6) is the clear winner by *both* best-case (1.29 m) and median
  (1.38 m) - not a single lucky config, a genuinely better architecture on
  this data. `siam2d_egomotion_fxb` (`TempoBandNet`) is a solid second
  (best 1.37 m, median 1.47 m, from fewer tested configs - see the testing
  issues below). The other ten architectures cluster tightly between 1.48
  and 1.80 m median, with no further standout; the two `StereoConv3DNet`
  window lengths (Sections 8-9) tie for the clear bottom - `c3d_3dhg_3_fxb`
  has the single worst median (1.80 m), `c3d_3dhg_10_fxb` the single worst
  best-case (1.76 m) - while `c3d_3dhg_10_fxb` alone is also by far the most
  expensive model to run (37 ms, 5x `siam2d_2dun_fxb`'s 6.8 ms, and 2.7x
  `c3d_3dhg_3_fxb`'s own 13.8 ms). Section 9's own question - "does
  stacking raw frames help at all, and how much of its extra cost is worth
  it" - gets a clean answer here: no, and none.
- **Learning rate**: `1e-3` and `3e-3` are both solid (mean 1.76 m / 1.67 m,
  best 1.30 m / 1.29 m); `1e-2` is measurably worse on every statistic
  (mean 1.90 m, worst outliers, and 8 of the round's 10 outright
  `overfitting`-flagged runs) - safe to keep excluding from the default
  grid, now confirmed across the full 12-model roster rather than the
  smaller set that first flagged it.
- **`fxb`-conditioning cost**: paired against its plain sibling at matching
  settings, every conditioned model lands within noise of its sibling
  (e.g. `siam2d_3dhg` 1.536/1.635 best/median vs `siam2d_3dhg_fxb`
  1.532/1.633) - taking `fxb` as an input costs nothing measurable on the
  one baseline this round ever tested it on. Whether it actually *earns*
  anything - generalizing to a baseline it never trained on - is a
  completely different, still-untested claim (Section 2.8).
- **Batch size**: no robust difference. Paired 8-vs-16 at matching
  model/lr/loss, batch 8 wins the simple count (37 of 63) but the median
  difference is +0.008 m - indistinguishable from noise at this scale.

**Architecture ranking by robustness, not just peak accuracy.** The
accuracy numbers above are best-case - the one config out of everything
tried that worked best. A different, equally useful question: handed this
architecture at a *random* point in the grid (any batch size, lr, or loss,
including the ones this round found were bad), how often does it come out
usable at all? Call a run **good** if its held-out depth MAE on the "same"
split is at or under 2.0 m - the natural break in the data: every
`logl1`/`hybrid` config from every architecture lands at or under 1.92 m,
and the next value up is 2.01 m, where the `smoothl1`/`depth26`/`1e-2`
failures start. Ranked by the fraction of each architecture's tested
configs that cleared that bar:

| Rank | Architecture | Good runs | Good % | Best MAE (m) | Median of good (m) | Worst MAE (m) | Median infer (ms) |
|---|---|---|---|---|---|---|---|
| 1 | `pyr2d_casc2d_fxb` | 13/15 | 87% | 1.49 | 1.62 | 3.76 | 7.6 |
| 2 | `yolo2d_3dhg_fxb` | 12/15 | 80% | 1.51 | 1.57 | 4.80 | 8.6 |
| 3 | `yolo2d_3dhg` | 13/17 | 76% | 1.48 | 1.60 | 8.40 | 8.6 |
| 3 | `mobile2d_3dhg` | 13/17 | 76% | 1.53 | 1.63 | 15.02 | 7.1 |
| 5 | `c3d_3dhg_10_fxb` | 11/15 | 73% | 1.76 | 1.78 | 2.93 | 37.2 |
| 6 | `mobile2d_3dhg_fxb` | 12/17 | 71% | 1.56 | 1.63 | 8.55 | 7.1 |
| 7 | `pyr2d_casc2d` | 11/16 | 69% | 1.50 | 1.61 | 4.76 | 7.5 |
| 8 | `siam2d_3dhg_fxb` | 11/17 | 65% | 1.53 | 1.63 | 4.96 | 7.4 |
| 9 | `siam2d_egomotion_fxb` | 6/11 | 55% | 1.37 | 1.47 | 5.05 | 30.4 |
| 10 | `c3d_3dhg_3_fxb` | 8/17 | 47% | 1.68 | 1.69 | 7.64 | 13.8 |
| 11 | `siam2d_3dhg` | 11/25 | 44% | 1.54 | 1.63 | 10.17 | 7.4 |
| 12 | `siam2d_2dun_fxb` | 6/15 | 40% | **1.29** | 1.30 | **22.96** | 6.8 |

The standout is the tension at the two ends: `siam2d_2dun_fxb` holds both
the single best accuracy in the entire round *and* the single worst (the
`smoothl1` divergence detailed in the red flags just below), and ranks dead
last on robustness as a direct result - it is the architecture most
sensitive to getting its
loss/lr picked correctly, not the safest default choice. `pyr2d_casc2d_fxb`
is the opposite: never the best, never close to the worst, usable under
almost any loss/lr thrown at it. `siam2d_3dhg` and `siam2d_3dhg_fxb` (44%,
65%) look worse here than their accuracy numbers alone suggest for exactly
the exploration-breadth reason flagged in the testing-process issues below
- `siam2d_3dhg` in particular carries every `depth26` and early `1e-2`
exploratory run this whole project ever tried (25 configs, more than any
other architecture), so a chunk of its "bad" count is inherited history,
not a property of the architecture itself. Read this table alongside the
accuracy ranking above, not instead of it: one says what the architecture
can do at its best, this says how much you have to get right to reach it.

*Red flags:*

- **`smoothl1` lets depth MAE drift away from the metric it's actually
  selected on.** `fit()` saves `best.pth` on `val_epe` improving, never on
  depth MAE (Section 14/15.1) - fine when the two track together, which
  `logl1`'s own log.csv shows them doing (depth MAE 1.48 m -> 1.34 m as EPE
  improves 3.04 -> 2.57 px, epochs 3 -> 19). Under `smoothl1` they can
  outright diverge: `siam2d_2dun_fxb bs8 lr3e-3 smoothl1` improves EPE
  every single evaluation (2.85 -> 2.52 px, epoch 3 -> 19) while its depth
  MAE gets **13x worse over the same epochs** (1.76 m -> 23.30 m) - and
  because `best.pth` only watches EPE, the checkpoint this run keeps is the
  epoch with the worst depth MAE it ever had. `mobile2d_3dhg bs16 lr3e-3
  smoothl1` shows the same shape (2.82 m -> 11.98 m while EPE keeps
  falling). This isn't "smoothl1 is a worse loss" restated - it's a
  specific flaw in what `best.pth` means for any disparity-only loss, on
  top of that.
- **Held-out environments disagree by metric, not just by model.** Gascola
  (the "different" split, Section 2.8) is systematically *easier* than
  SeasonalForestAutumn (the "same" split) by EPE - mean/median about 0.57
  px lower across all 197 rows - but is a coin flip on depth MAE (lower in
  94 of 197, median difference +0.02 m). An "it generalizes well" claim
  that only checks EPE on a new environment could be measuring an easier
  environment, not a better model.
- **High learning rates fail in more than one way.** Beyond the `1e-2`
  configs that simply score worse, two `yolo2d_3dhg`/`yolo2d_3dhg_fxb` runs
  at `1e-2`/`hybrid` locked in a reasonable `best.pth` early (epoch 11 and
  3) and then diverged to `NaN` later in the same run (summary.csv's
  `final_val_epe` for both is literally `nan`) - survivable only because
  checkpoint selection happened to catch them before the blowup. Several
  `siam2d_2dun_fxb` runs at `1e-2` (and some at `3e-3`) weren't so lucky:
  they went `NaN` from epoch 0 and never produced a usable `best.pth` at
  all. Those run directories have since been deleted as unrecoverable, so
  they no longer appear in `eval_all.csv` - but they happened, at an
  8-out-of-8 rate for this architecture at `1e-2`.

*Configs now safe to exclude from future sweeps:*

- `--grid-loss smoothl1 depth26` - already off by default; this round's
  full-scale numbers (and the divergence red flag above) are a second,
  stronger confirmation, not a new finding.
- `--grid-lr 1e-2` - already off by default; same situation, now confirmed
  across all 12 models instead of the 8 that first flagged it.
- Both `StereoConv3DNet` window lengths as a priority architecture - between
  them they hold the worst median (`c3d_3dhg_3_fxb`) and worst best-case
  (`c3d_3dhg_10_fxb`) accuracy of anything tested, and `c3d_3dhg_10_fxb` is
  also by far the most expensive model to run; keep both as a documented
  negative result rather than spend more sweep budget on them.
- `siam2d_2dun_fxb` + `smoothl1` specifically - the single worst number in
  the entire dataset (22.96 m) landed on the round's *best* architecture,
  paired with the one loss already excluded on other grounds. Not a reason
  to doubt the architecture; a concrete reason never to pair it with
  `smoothl1` again.

*Must still be tested:*

- **A noise floor.** Zero of the 197 configs share a (model, batch size,
  lr, loss) with a different seed. The 0.03-0.1 m gaps separating the top
  5-10 configs could be real or could be exactly the kind of run-to-run
  variation `set_seed()`'s own docstring (`train_tartanair.py`) warns GPU
  runs still have, even seeded identically: bilinear interpolate's backward
  pass and the cost-volume scatter both accumulate with `atomicAdd`, whose
  summation order isn't fixed run to run. Re-running the top handful of
  configs at 2-3 extra seeds is cheap and was proposed earlier in this
  project's own history but never
  actually run.
- **`lr=3e-4`** has been tried on exactly one of twelve architectures
  (`siam2d_2dun_fxb`, where it ties or slightly beats `1e-3`/`3e-3`) -
  whether going below `1e-3` helps *any other* architecture is completely
  unknown. Cheapest next experiment: add `3e-4` to the default `--grid-lr`
  for the other eleven.
- **The cross-baseline generalization test (Section 2.8)** - the actual
  reason four architectures carry `fxb` conditioning. This round establishes
  it's free; it does not establish it works.
- **Only 2 of TartanAir's 5 environments have ever been a validation
  target** across all 197 runs: `SeasonalForestAutumn` ("same") and
  `Gascola` ("different", Section 2.8's `alt_environment_split`, fixed at
  `--diff-seed 1337`). `AbandonedFactory`, `Downtown`, and `Office` have
  never been held out once. Every generalization claim in this section
  rests on a 2-environment sample out of 5 available.
- **A full, properly-trained run.** 147 of the 197 configs (75%) are
  flagged `"still improving at cutoff"` by `eval_all.py`'s own
  `log_csv_fit_stats()` - meaning most of this round's
  rankings compare models that hadn't finished improving when the 20-epoch
  budget ran out, concentrated exactly where expected (57/72 at `lr=1e-3`,
  45/64 at `lr=3e-3` - the lower, still-annealing rates; `overfitting` is
  instead concentrated at `1e-2`, 8/45). The *relative* ranking above is
  reasonably trustworthy for that reason - unstable configs failed loudly,
  stable ones mostly hadn't converged yet in the same direction - but the
  absolute numbers, and possibly the order of the tightly-clustered middle
  ten architectures, could still move once the winner is trained properly
  (the `--data-fraction 1.0`, up to 100 epochs, early-stopping run `report.md`
  already generates a command for - Section 15.3's `write_report`).

*Issues with the testing process itself:*

- **Uneven exploration breadth per architecture skews any naive "median
  over all its tested configs" comparison.** `siam2d_3dhg` and
  `siam2d_egomotion_fxb` were the two models present in the very first,
  most exploratory sweep (`depth26`, `1e-2`, before either was known to be
  bad) and so each accumulated more bad-config rows than the other ten
  architectures added later only under already-pruned defaults. Comparing
  medians across *all* tested configs (rather than the `logl1`/`hybrid`-only
  numbers used above) would unfairly penalize exactly the two models that
  were explored the most thoroughly.
- **`eval_all.csv`'s own numbers carry a second, independent noise source**
  on top of training-time variance: each split is capped to
  `--max-eval-windows` (200 by default) for a tractable full-grid scan,
  and the dataset is built with `augment=True` (matching how this project's
  own training/validation has always worked, Section 2.7) - so even
  re-evaluating the exact same checkpoint twice would give slightly
  different numbers from the random crop alone, something the top-15
  table's tight 0.01-0.03 m spreads could already be partly made of.
- **`avg_infer_ms` in `eval_all.csv` is a batched-GPU number** (default
  `--eval-bs 8`, run_window on whatever GPU `eval_all.py` ran on), not the
  single-image, half-precision, target-hardware number Section 1.1's
  actual requirement cares about. `siam2d_2dun_fxb` (3.79 M params) and
  `siam2d_3dhg` (0.20 M params) measure almost identically here (6.8 ms vs
  7.4 ms) despite a 19x parameter gap - a reminder that these numbers say
  "cost on this GPU, this way of batching" and nothing yet about the
  Jetson-class edge target `evaluate.py --bench` is built to measure
  (Section 15.2).

#### 15.4 Command-line reference (`train_tartanair.py`)

`train.py` takes the same core arguments; the TartanAir-specific and sweep
arguments below are unique to `train_tartanair.py`.

| argument | default | purpose |
|---|---|---|
| `--data` | *required* | TartanAir folder(s); finds all trajectories inside automatically |
| `--model` | `siam2d_3dhg` | `siam2d_3dhg`, `siam2d_3dhg_fxb`, `siam2d_2dun_fxb`, `c3d_3dhg_10_fxb`, `c3d_3dhg_3_fxb`, `siam2d_egomotion_fxb`, `mobile2d_3dhg`, `mobile2d_3dhg_fxb`, `pyr2d_casc2d`, `pyr2d_casc2d_fxb`, `yolo2d_3dhg`, `yolo2d_3dhg_fxb` (Sections 6-10, Section 13's table has the flag-to-architecture mapping) |
| `--loss` | `smoothl1` | `smoothl1`, `depth26`, `logl1`, `hybrid` (Section 14) |
| `--loss-w-log`, `--loss-w-grad` | 1.0, 0.5 | how much weight to give the log-space term and the edge-matching term |
| `--loss-silog-lambda` | 1.0 | how scale-invariant `depth26` is; 0.0 keeps absolute scale instead |
| `--yolo-scale` | `n` | YOLO26 model size for `--model yolo2d_3dhg` (`n`: 2.61 M params, `s`: 9.39 M params) |
| `--crop` | none | `HxW` random crop, must be a multiple of 16 - **32 for `--model yolo2d_3dhg`**. `480x640` works for both and matches the real camera's aspect ratio |
| `--max-disp` | 128 | how far the disparity search reaches; caps how close an object can be before depth clips |
| `--scale` | 1.0 | image resize factor; scales the camera's focal length and disparity to match (Section 2.7) |
| `--camera` | `front` | which V2 camera rig to use; only `front` is checked against this project's pose convention |
| `--difficulty`, `--envs` | both, all | limit to `Data_easy`/`Data_hard`, or to specific named environments |
| `--val-envs`, `--val-frac` | last 15% | which environments are held out for validation (Section 2.7) |
| `--window`, `--frame-stride` | model-dependent, 1 | how many frames per window (1 single-frame, 4 `siam2d_egomotion_fxb`, 10 `c3d_3dhg_10_fxb`, 3 `c3d_3dhg_3_fxb`) and the gap between them |
| `--pose-noise`, `--pose-noise-trans` | 0.0, 0.02 | how much fake motion-sensor noise to add during training (`siam2d_egomotion_fxb`) |
| `--max-train-windows`, `--max-val-windows` | 1500, 1000 | cap on how much data each sweep cell uses, so cells are comparable |
| `--eval-every`, `--print-every` | 2, 10 | how often to validate; how often to print progress |
| `--amp`, `--workers` | off, 4 | mixed precision; number of data-loading workers |
| `--resume` / `--init` | none | continue a run (weights + optimiser + epoch) / start from these weights only |
| `--out`, `--log-dir` | `runs/tartanair`, `logs` | where this run is saved; folder collecting one log file per run |
| `--grid-search` | off | turn on the sweep described in Section 15.3 |
| `--grid-models`, `--grid-bs`, `--grid-lr`, `--grid-loss` | all 12 `--model` choices, `8 16`, `1e-3 3e-3`, `logl1 hybrid` | the four things the sweep tries every combination of (Section 15.3); `depth26` and `smoothl1` are off by default because sweeps measured them and found them clearly worse (pass either explicitly to include it anyway) |
| `--grid-epochs` | 10 | epochs per combination, kept separate from `--epochs` |
| `--grid-force-rerun` | off | redo combinations an earlier sweep already finished |
| `--no-tensorboard`, `--tensorboard-port` | on, 6006 | the training dashboard for the sweep |

#### 15.5 Repository map

| file | role |
|---|---|
| [collect_dataset.py](collect_dataset.py) | record rides from AirSim (Section 2) |
| [stereo_datasets.py](stereo_datasets.py) | window assembly, disparity GT, pose chain, TartanAir loader (Section 2.5) |
| [models/common.py](models/common.py) | shared blocks: encoder, cost volume, 3D hourglass, refinement |
| [models/stereo_conv_net.py](models/stereo_conv_net.py) | `StereoConvNet`: 1D correlation, 2D-conv U-Net aggregation, learned upsample (Section 6) |
| [models/baseline_net.py](models/baseline_net.py) | `FastStereoNet` (Section 7) |
| [models/stereo_conv3d_net.py](models/stereo_conv3d_net.py) | `StereoConv3DNet`: 3D-conv encoder + `FastStereoNet`'s aggregation/refinement, `c3d_3dhg_10_fxb` (T=10, Section 8) and `c3d_3dhg_3_fxb` (T=3, Section 9) share this one class |
| [models/temporal_net.py](models/temporal_net.py) | `TempoBandNet`: splatting, gate, narrow bands (Section 11) |
| [models/yolo_stereo.py](models/yolo_stereo.py) | `YoloStereoNet`: YOLO26 encoder/decoder around the shared cost volume |
| [models/yolo26_stereo.yaml](models/yolo26_stereo.yaml) | vendored YOLO26 backbone + PAN neck config for the above |
| [losses.py](losses.py) | the four `--loss` objectives (Section 14) |
| [train.py](train.py) | training loop, windows, pose noise, sequence-split validation (Section 15.1) |
| [evaluate.py](evaluate.py) | metrics per depth range + latency benchmark (Section 15.2) |

---

### 16. Bibliography

Grouped by section; every entry includes a BibTeX record and a URL.

#### Pre-2020 - Stereo Triangulation

```bibtex
@article{scharstein2002taxonomy,
  title   = {A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms},
  author  = {Scharstein, Daniel and Szeliski, Richard},
  journal = {International Journal of Computer Vision},
  volume  = {47},
  number  = {1-3},
  pages   = {7--42},
  year    = {2002},
  doi     = {10.1023/A:1014573219977},
  url     = {https://link.springer.com/article/10.1023/A:1014573219977}
}

@article{hirschmuller2008sgm,
  author  = {Hirschm{\"u}ller, Heiko},
  title   = {Stereo Processing by Semi-Global Matching and Mutual Information},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {30},
  number  = {2},
  pages   = {328--341},
  year    = {2008},
  doi     = {10.1109/TPAMI.2007.1166},
  url     = {https://ieeexplore.ieee.org/document/4359315}
}

@article{zbontar2016mccnn,
  title   = {Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches},
  author  = {{\v{Z}}bontar, Jure and LeCun, Yann},
  journal = {Journal of Machine Learning Research},
  volume  = {17},
  pages   = {1--32},
  year    = {2016},
  eprint  = {1510.05970},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/1510.05970}
}

@inproceedings{mayer2016sceneflow,
  title     = {A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation},
  author    = {Mayer, Nikolaus and Ilg, Eddy and H{\"a}usser, Philip and Fischer, Philipp and Cremers, Daniel and Dosovitskiy, Alexey and Brox, Thomas},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016},
  eprint    = {1512.02134},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/1512.02134}
}

@inproceedings{kendall2017gcnet,
  title     = {End-to-End Learning of Geometry and Context for Deep Stereo Regression},
  author    = {Kendall, Alex and Martirosyan, Hayk and Dasgupta, Saumitro and Henry, Peter},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year      = {2017},
  eprint    = {1703.04309},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/1703.04309}
}

@inproceedings{chang2018psmnet,
  title     = {Pyramid Stereo Matching Network},
  author    = {Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018},
  url       = {https://openaccess.thecvf.com/content_cvpr_2018/html/Chang_Pyramid_Stereo_Matching_CVPR_2018_paper.html}
}
```

#### Pre-2020 - Monocular Depth Estimation

```bibtex
@article{saxena2009make3d,
  title   = {Make3D: Learning 3D Scene Structure from a Single Still Image},
  author  = {Saxena, Ashutosh and Sun, Min and Ng, Andrew Y.},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {31},
  number  = {5},
  pages   = {824--840},
  year    = {2009},
  url     = {https://www.cs.cornell.edu/~asaxena/reconstruction3d/saxena_make3d_learning3dstructure.pdf}
}

@inproceedings{eigen2014depth,
  title     = {Depth Map Prediction from a Single Image using a Multi-Scale Deep Network},
  author    = {Eigen, David and Puhrsch, Christian and Fergus, Rob},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  pages     = {2366--2374},
  year      = {2014},
  url       = {https://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network}
}

@inproceedings{laina2016fcrn,
  title     = {Deeper Depth Prediction with Fully Convolutional Residual Networks},
  author    = {Laina, Iro and Rupprecht, Christian and Belagiannis, Vasileios and Tombari, Federico and Navab, Nassir},
  booktitle = {Fourth International Conference on 3D Vision (3DV)},
  pages     = {239--248},
  year      = {2016},
  doi       = {10.1109/3DV.2016.32},
  eprint    = {1606.00373},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/1606.00373}
}

@inproceedings{godard2017monodepth,
  title     = {Unsupervised Monocular Depth Estimation with Left-Right Consistency},
  author    = {Godard, Cl{\'e}ment and Mac Aodha, Oisin and Brostow, Gabriel J.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {6602--6611},
  year      = {2017},
  url       = {https://openaccess.thecvf.com/content_cvpr_2017/html/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.html}
}

@inproceedings{fu2018dorn,
  title     = {Deep Ordinal Regression Network for Monocular Depth Estimation},
  author    = {Fu, Huan and Gong, Mingming and Wang, Chaohui and Batmanghelich, Kayhan and Tao, Dacheng},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018},
  url       = {https://openaccess.thecvf.com/content_cvpr_2018/html/Fu_Deep_Ordinal_Regression_CVPR_2018_paper.html}
}

@inproceedings{godard2019monodepth2,
  title     = {Digging Into Self-Supervised Monocular Depth Estimation},
  author    = {Godard, Cl{\'e}ment and Mac Aodha, Oisin and Firman, Michael and Brostow, Gabriel J.},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {3827--3837},
  year      = {2019},
  doi       = {10.1109/ICCV.2019.00393},
  url       = {https://openaccess.thecvf.com/content_ICCV_2019/html/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.html}
}
```

#### Datasets

```bibtex
@inproceedings{geiger2012kitti,
  title     = {Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  author    = {Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3354--3361},
  year      = {2012},
  url       = {https://www.cvlibs.net/publications/Geiger2012CVPR.pdf}
}

@inproceedings{menze2015kitti15,
  title     = {Object Scene Flow for Autonomous Vehicles},
  author    = {Menze, Moritz and Geiger, Andreas},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3061--3070},
  year      = {2015},
  url       = {https://www.cvlibs.net/publications/Menze2015CVPR_abstract.pdf}
}

@inproceedings{scharstein2014middlebury,
  title     = {High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth},
  author    = {Scharstein, Daniel and Hirschm{\"u}ller, Heiko and Kitajima, York and Krathwohl, Greg and Nesic, Nera and Wang, Xi and Westling, Porter},
  booktitle = {German Conference on Pattern Recognition (GCPR)},
  series    = {Lecture Notes in Computer Science},
  volume    = {8753},
  pages     = {31--42},
  year      = {2014},
  url       = {https://www.cs.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf}
}

@inproceedings{schops2017eth3d,
  title     = {A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos},
  author    = {Sch{\"o}ps, Thomas and Sch{\"o}nberger, Johannes L. and Galliani, Silvano and Sattler, Torsten and Schindler, Konrad and Pollefeys, Marc and Geiger, Andreas},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2017},
  url       = {https://openaccess.thecvf.com/content_cvpr_2017/html/Schops_A_Multi-View_Stereo_CVPR_2017_paper.html}
}

@inproceedings{yang2019drivingstereo,
  title     = {DrivingStereo: A Large-Scale Dataset for Stereo Matching in Autonomous Driving Scenarios},
  author    = {Yang, Guorun and Song, Xiao and Huang, Chaoqin and Deng, Zhidong and Shi, Jianping and Zhou, Bolei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {899--908},
  year      = {2019},
  url       = {https://drivingstereo-dataset.github.io/}
}

@inproceedings{silberman2012nyuv2,
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  author    = {Silberman, Nathan and Hoiem, Derek and Kohli, Pushmeet and Fergus, Rob},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2012},
  url       = {https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html}
}

@inproceedings{wang2020tartanair,
  title     = {TartanAir: A Dataset to Push the Limits of Visual SLAM},
  author    = {Wang, Wenshan and Zhu, Delong and Wang, Xiangwei and Hu, Yaoyu and Qiu, Yuheng and Wang, Chen and Hu, Yafei and Kapoor, Ashish and Scherer, Sebastian},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2020},
  doi       = {10.1109/IROS45743.2020.9341801},
  eprint    = {2003.14338},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2003.14338}
}

@inproceedings{fonder2019midair,
  title     = {Mid-Air: A Multi-Modal Dataset for Extremely Low Altitude Drone Flights},
  author    = {Fonder, Michael and Van Droogenbroeck, Marc},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages     = {553--562},
  year      = {2019},
  url       = {https://midair.ulg.ac.be/}
}

@article{zhang2023uavstereo,
  title   = {UAVStereo: A Multiple Resolution Dataset for Stereo Matching in UAV Scenarios},
  author  = {Zhang, Xiaoyi and Cao, Xuefeng and Yu, Anzhu and Yu, Wenshuai and Li, Zhenqi and Quan, Yujun},
  journal = {arXiv preprint},
  year    = {2023},
  eprint  = {2302.10082},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2302.10082}
}
```

#### Recent SoA (last 3 years)

```bibtex
@inproceedings{xu2023igev,
  title     = {Iterative Geometry Encoding Volume for Stereo Matching},
  author    = {Xu, Gangwei and Wang, Xianqi and Ding, Xiaohuan and Yang, Xin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {21919--21928},
  year      = {2023},
  eprint    = {2303.06615},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2303.06615}
}

@inproceedings{zhang2023temporalstereo,
  title     = {TemporalStereo: Efficient Spatial-Temporal Stereo Matching Network},
  author    = {Zhang, Youmin and Poggi, Matteo and Tosi, Fabio and Mattoccia, Stefano},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {9528--9535},
  year      = {2023},
  eprint    = {2211.13755},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2211.13755}
}

@inproceedings{zeng2024tcstereo,
  title     = {Temporally Consistent Stereo Matching},
  author    = {Zeng, Jiaxi and Yao, Chengtang and Wu, Yuwei and Jia, Yunde},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2024},
  eprint    = {2407.11950},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2407.11950}
}

@inproceedings{jing2025stereoanyvideo,
  title     = {Stereo Any Video: Temporally Consistent Stereo Matching},
  author    = {Jing, Junpeng and Luo, Weixun and Mao, Ye and Mikolajczyk, Krystian},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  eprint    = {2503.05549},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2503.05549}
}

@inproceedings{guo2025lightstereo,
  title     = {LightStereo: Channel Boost Is All You Need for Efficient 2D Cost Aggregation},
  author    = {Guo, Xianda and Zhang, Chenming and Zhang, Youmin and Zheng, Wenzhao and Nie, Dujun and Poggi, Matteo and Chen, Long},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025},
  eprint    = {2406.19833},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2406.19833}
}

@article{tahmasebi2025esmstereo,
  title   = {ESMStereo: Enhanced ShuffleMixer Disparity Upsampling for Real-Time and Accurate Stereo Matching},
  author  = {Tahmasebi, Mahmoud and Huq, Saif and Meehan, Kevin and McAfee, Marion},
  journal = {arXiv preprint},
  year    = {2025},
  eprint  = {2506.21091},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2506.21091}
}

@article{liang2026psknet,
  title   = {PSKNet: Lightweight Kernel-Aware Slice Network for Real-Time Stereo Depth Estimation on Edge Devices},
  author  = {Liang, Bifa and Hu, Ziyang and Hu, Haifeng and Xu, Jianming and Chen, Dihu},
  journal = {Knowledge-Based Systems},
  volume  = {336},
  year    = {2026},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S0950705126000535},
  note    = {Measured on Jetson AGX Orin - the same edge-device class this project targets (Section 1.1)}
}

@article{wen2026fastfoundationstereo,
  title   = {Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching},
  author  = {Wen, Bowen and Dewan, Shaurya and Birchfield, Stan},
  journal = {arXiv preprint},
  year    = {2025},
  eprint  = {2512.11130},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2512.11130},
  note    = {NVIDIA; distills/prunes FoundationStereo to 21 ms on an RTX 3090 with TensorRT}
}
```

#### YOLO Lineage (Section 12)

```bibtex
@inproceedings{redmon2016yolo,
  title     = {You Only Look Once: Unified, Real-Time Object Detection},
  author    = {Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {779--788},
  year      = {2016},
  doi       = {10.1109/CVPR.2016.91},
  eprint    = {1506.02640},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/1506.02640},
  note      = {Origin of the single-stage detection paradigm the YOLO26 backbone/neck
               (Section 12) descends from; ten releases and nine years removed from
               the architecture YoloStereoNet (Section 10.3) actually borrows}
}
```

#### Miscellaneous (referenced in Section 1 / Section 5)

```bibtex
@article{min2014fastglobalsmoothing,
  title   = {Fast Global Image Smoothing Based on Weighted Least Squares},
  author  = {Min, Dongbo and Choi, Sunghwan and Lu, Jiangbo and Ham, Bumsub and Sohn, Kwanghoon and Do, Minh N.},
  journal = {IEEE Transactions on Image Processing},
  volume  = {23},
  number  = {12},
  pages   = {5638--5653},
  year    = {2014},
  doi     = {10.1109/TIP.2014.2366600},
  url     = {https://docs.opencv.org/4.x/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html},
  note    = {Algorithmic basis of OpenCV's \texttt{cv2.ximgproc.createDisparityWLSFilter}, used as \texttt{STEREO\_ENHANCE\_WLS} in this project's config.py}
}

@article{yang2024depthanythingv2,
  title   = {Depth Anything V2},
  author  = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal = {arXiv preprint},
  year    = {2024},
  eprint  = {2406.09414},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2406.09414}
}

@inproceedings{sandler2018mobilenetv2,
  title     = {MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author    = {Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4510--4520},
  year      = {2018},
  doi       = {10.1109/CVPR.2018.00474},
  note      = {Inverted-residual block used by MobileStereoNet (Section 10.1)}
}

@inproceedings{wang2019anytime,
  title     = {Anytime Stereo Image Depth Estimation on Mobile Devices},
  author    = {Wang, Yan and Lai, Zihang and Huang, Gao and Wang, Brian H. and van der Maaten, Laurens and Campbell, Mark and Weinberger, Kilian Q.},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {5893--5900},
  year      = {2019},
  doi       = {10.1109/ICRA.2019.8794003},
  note      = {Coarse-to-fine residual disparity search followed by AnyStereoNet (Section 10.2)}
}

@misc{ultralytics2025depth,
  title  = {Monocular Depth Estimation with Ultralytics YOLO26},
  author = {{Ultralytics}},
  year   = {2025},
  url    = {https://docs.ultralytics.com/tasks/depth},
  note   = {Software documentation. No accompanying paper, and neither the
            documentation nor the release announcement cites prior work; the
            method's components are attributable to \texttt{eigen2014depth}
            (SILog and the $\delta_n$ metrics), \texttt{ranftl2020midas}
            (multi-scale gradient matching) and \texttt{yang2024depthanythingv2}
            (mixed-dataset scale-invariant training, pseudo-labelled unlabelled
            images). Source inspected at v8.4.130, AGPL-3.0}
}

@article{ranftl2020midas,
  title   = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
  author  = {Ranftl, Ren\'e and Lasinger, Katrin and Hafner, David and Schindler, Konrad and Koltun, Vladlen},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {44},
  number  = {3},
  pages   = {1623--1637},
  year    = {2022},
  doi     = {10.1109/TPAMI.2020.3019967},
  note    = {Origin of the multi-scale gradient-matching loss term used in losses.py}
}
```
