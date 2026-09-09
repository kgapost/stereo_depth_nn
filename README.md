# Stereo Depth NN

Research code that trains a neural network to guess **depth** (how far away things are) from a **stereo camera pair** (two cameras, side by side, like human eyes).

This is for a drone project. The drone needs to know how far obstacles are, so it can avoid them. The full drone project lives in a separate repo, [obstacle_detection_sim](https://github.com/kgapost/obstacle-detection-sim). This repo is only for the AI research part: training and testing new depth models. It is not needed to fly the drone.

## Why train a neural network for this?

The drone project already computes depth with a normal (non-AI) method, called **stereo block matching**. It works, but it has three problems:

1. **Holes.** Some pixels get no answer at all - for example a flat wall, a shiny surface, or the sky. The old method just drops these pixels instead of filling them in.
2. **No smart use of motion.** The old method smooths the depth map over time, but it does not know the drone is moving. This turns into blur and lag while flying.
3. **Hand-tuned, not learned.** The old method needs its settings tuned by a person. A trained network can learn good settings from data instead.

This repo trains and tests different neural networks that try to fix these problems. The training data comes from the AirSim simulator, not a real drone.

## What is in this repo

| File / folder | What it does |
|---|---|
| `collect_dataset.py` | Records left camera, right camera, and the true depth from AirSim. Saves them to disk as a dataset. |
| `debug_depth_capture.py` | A small test tool. Checks which AirSim depth-capture method works, without needing to fly. |
| `run_dataset_campaign.sh` | Runs `collect_dataset.py` many times in a row - once per world / weather / speed combination. |
| `stereo_datasets.py` | Reads a saved dataset (or the public TartanAir dataset) and prepares it for training. |
| `models/` | The neural network models themselves (see the table below). |
| `losses.py` | The different training loss functions you can pick with `--loss`. |
| `train.py` | Trains one model on a dataset you collected with `collect_dataset.py`. |
| `train_tartanair.py` | Trains one model (or many, in a "grid search") on the public TartanAir dataset. |
| `evaluate.py` | Checks how good a trained model is, and measures how fast it runs. |
| `config.py`, `utils_airsim.py`, `json_templates/` | Copied from the drone repo. These start AirSim and read its camera settings. Needed only by `collect_dataset.py` and `debug_depth_capture.py` - the training/evaluation files above never touch them. |

## Setup

**1. AirSim must already be installed and able to run.** This repo does not install AirSim - it only talks to it. For the full install steps (Unreal Engine, AirSim, the AirSim worlds used below), see [obstacle_detection_sim](https://github.com/kgapost/obstacle-detection-sim)'s README.md, sections 3.1 and 2.3.1.

**2. Create a Python environment just for this repo:**
```bash
python3 -m venv stereo_env
./stereo_env/bin/pip install -r requirements.txt
```
This is a **separate, smaller** environment than the drone project's. It only needs PyTorch, OpenCV, NumPy, and (for the `yolo` model only) `ultralytics` - no ROS.

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

## Step 2 - Train a model

Eight models are available. All of them are already built and working - none are just ideas on paper:

| `--model` | What it does, in simple words |
|---|---|
| `stereoconv` | Looks at one image pair. Simple and easy to follow - the slow starting point. |
| `baseline` | Looks at one image pair. Small and fast - built to run on the drone's onboard computer. |
| `stereoconv3d` | Looks at the last 10 frames together. Uses time, but keeps no memory between runs. |
| `stereoconv3d_fast` | Same as `stereoconv3d`, but only looks at 3 frames instead of 10 - cheaper. |
| `mobilenet` | Same idea as `baseline`, but built from smaller, cheaper pieces. The smallest model here. |
| `anynet` | Same idea as `baseline`, but gives a quick rough answer first, then improves it if there is time to spare. |
| `yolo` | Same idea as `baseline`, but uses a much bigger, stronger image feature extractor. Built to see further away. |
| `temporal` | Remembers the past. Uses the drone's own motion to guess where things moved. The strongest model here so far. |

Train one model on data you collected in Step 1:
```bash
python train.py --model baseline --data ~/datasets/airsim_stereo --out runs/baseline
```

Or train on the public TartanAir dataset instead:
```bash
python train_tartanair.py --model temporal --data ~/datasets/tartanair --out runs/temporal
```

Both scripts take many more options, for example `--loss` (which training objective to use, see `losses.py`), `--bs` (batch size), and `--epochs`. To see the full list:
```bash
python train.py --help
python train_tartanair.py --help
```

### Try many settings at once (grid search)

`train_tartanair.py --grid-search` trains many combinations of model, batch size, learning rate, and loss function automatically, and writes one `summary.csv` file comparing all of them:

```bash
python train_tartanair.py --grid-search --data ~/datasets/tartanair \
    --grid-models baseline temporal mobilenet anynet \
    --out runs/my_grid_search
```

Each combination trains as its own process, so one crashing (for example, running out of GPU memory) does not stop the others - it is just marked `FAILED` in the summary. If the sweep is stopped (Ctrl+C, or it crashes) and started again with the same command, finished combinations are skipped instead of redone.

## Step 3 - Check how good a trained model is

```bash
python evaluate.py --ckpt runs/baseline/best.pth --model baseline --data ~/datasets/airsim_stereo_test
```

This reports how far off the depth guess is - in pixels and in metres, split by distance (close / medium / far, since the same pixel error means a much bigger real-world error far away than it does up close). It can also measure how fast the model runs, e.g. on a Jetson:
```bash
python evaluate.py --model baseline --bench --size 256x448 --fp16
```

## A note on camera settings

The real drone's camera is a Waveshare/Seeed IMX219-83, with a **6 cm** distance between its two lenses (the "baseline") and an **83°** field of view. `json_templates/settings_dataset.json` is already set up to match this, so depth recorded in AirSim lines up with what the real camera would see. If you change the real camera, update that file to match.

## Where to learn more

This README covers how to run the code. The deeper research write-up (why each model is designed this way, the literature review, and the full results) is kept as private notes, not in this repo - ask the repo owner if you need it.
