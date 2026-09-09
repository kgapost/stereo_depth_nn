"""PyTorch datasets for stereo depth training.

Two loaders share one sample format:
  - AirSimStereoDataset : sequences recorded by collect_dataset.py
  - TartanAirDataset    : the public TartanAir benchmark (also AirSim-generated),
                          for reproducible evaluation / extra training data.

A sample is a dict of tensors:
  left, right : (T, 3, H, W) float in [0, 1]   (T = temporal window, 1 for baseline)
  disp        : (T, 1, H, W) ground-truth disparity in pixels (left view)
  valid       : (T, 1, H, W) 1 where the disparity is supervisable
  rel_pose    : (T, 4, 4) T_cam_t<-cam_{t-1} in the *optical* frame
                (x right, y down, z forward); identity at t=0
  first       : (T,) 1.0 for the first frame of the window (no temporal prior)
  K           : (4,) fx, fy, cx, cy of the (possibly cropped) full-res image
  fxb         : ()  fx * baseline, so depth = fxb / disp

AirSim poses are stored in world NED with the camera frame x=forward y=right
z=down; TartanAir uses the same convention. Both are converted to the optical
frame here so the models never see NED.
"""

import csv
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

# Permutation NED-style camera frame (x fwd, y right, z down) -> optical
# (x right, y down, z fwd): v_opt = P @ v_cam.
_P_CAM2OPT = np.array([[0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0]])


def quat_to_rot(qw, qx, qy, qz):
    """Rotation matrix from a (w, x, y, z) quaternion (maps body -> world)."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def world_T_optical(pos, quat_wxyz):
    """4x4 world-from-optical-camera transform from an AirSim/TartanAir pose."""
    T = np.eye(4)
    R_w_cam = quat_to_rot(*quat_wxyz)          # camera(NED-style) -> world
    T[:3, :3] = R_w_cam @ _P_CAM2OPT.T         # optical -> world
    T[:3, 3] = pos
    return T


def relative_pose(T_w_prev, T_w_cur):
    """T_cur<-prev = inv(T_w_cur) @ T_w_prev (both optical)."""
    return np.linalg.inv(T_w_cur) @ T_w_prev


class _StereoSequenceDataset(Dataset):
    """Common machinery: windows over sequences, cropping, augmentation."""

    def __init__(self, window=1, frame_stride=1, window_stride=None,
                 crop=None, augment=False, max_disp=128, max_depth=95.0):
        self.window = window
        self.frame_stride = frame_stride
        self.window_stride = window_stride or max(1, window // 2)
        self.crop = crop  # (H, W) or None
        self.augment = augment
        self.max_disp = max_disp
        self.max_depth = max_depth
        # index = list of (sequence_id, [frame indices of the window])
        self.index = []
        self.sequences = []  # per-seq dict: frames, calib, poses (T_w_opt list)

    def _build_index(self):
        for sid, seq in enumerate(self.sequences):
            n = len(seq["frames"])
            span = (self.window - 1) * self.frame_stride + 1
            for start in range(0, n - span + 1, self.window_stride):
                idxs = [start + k * self.frame_stride for k in range(self.window)]
                self.index.append((sid, idxs))

    def __len__(self):
        return len(self.index)

    # -- to be provided by subclasses -----------------------------------
    def _load_frame(self, seq, i):
        """Return (left_bgr uint8, right_bgr uint8, depth float32 meters)."""
        raise NotImplementedError

    # --------------------------------------------------------------------
    def __getitem__(self, item):
        sid, idxs = self.index[item]
        seq = self.sequences[sid]
        calib = seq["calib"]
        fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
        fxb = calib["fx"] * calib["baseline_m"]

        lefts, rights, disps, valids, rels, firsts = [], [], [], [], [], []
        x0 = y0 = 0
        jitter = None
        for k, i in enumerate(idxs):
            left, right, depth = self._load_frame(seq, i)
            H, W = depth.shape

            if self.crop is not None:
                ch, cw = self.crop
                if k == 0:
                    y0 = np.random.randint(0, H - ch + 1) if self.augment else (H - ch) // 2
                    x0 = np.random.randint(0, W - cw + 1) if self.augment else (W - cw) // 2
                left = left[y0:y0 + ch, x0:x0 + cw]
                right = right[y0:y0 + ch, x0:x0 + cw]
                depth = depth[y0:y0 + ch, x0:x0 + cw]

            valid = (depth > 0.2) & (depth < self.max_depth)
            disp = np.where(valid, fxb / np.maximum(depth, 1e-3), 0.0).astype(np.float32)
            valid &= disp < self.max_disp

            left = left.astype(np.float32) / 255.0
            right = right.astype(np.float32) / 255.0
            if self.augment:
                if jitter is None:  # one photometric jitter per window (consistent in time)
                    jitter = (np.random.uniform(0.8, 1.2),          # brightness
                              np.random.uniform(0.8, 1.2),          # contrast
                              np.random.uniform(-0.05, 0.05, (1, 1, 3)))  # color shift
                b, c, shift = jitter
                for img in (left, right):
                    np.multiply(img, b, out=img)
                    np.add((img - 0.5) * c + 0.5, shift, out=img)
                np.clip(left, 0, 1, out=left)
                np.clip(right, 0, 1, out=right)

            # BGR (cv2) -> RGB, HWC -> CHW
            lefts.append(torch.from_numpy(np.ascontiguousarray(left[..., ::-1].transpose(2, 0, 1))))
            rights.append(torch.from_numpy(np.ascontiguousarray(right[..., ::-1].transpose(2, 0, 1))))
            disps.append(torch.from_numpy(disp).unsqueeze(0))
            valids.append(torch.from_numpy(valid.astype(np.float32)).unsqueeze(0))

            if k == 0:
                rels.append(torch.eye(4))
                firsts.append(1.0)
            else:
                T_rel = relative_pose(seq["poses"][idxs[k - 1]], seq["poses"][i])
                rels.append(torch.from_numpy(T_rel.astype(np.float32)))
                firsts.append(0.0)

        return {
            "left": torch.stack(lefts),
            "right": torch.stack(rights),
            "disp": torch.stack(disps),
            "valid": torch.stack(valids),
            "rel_pose": torch.stack(rels),
            "first": torch.tensor(firsts, dtype=torch.float32),
            "K": torch.tensor([fx, fy, cx - x0, cy - y0], dtype=torch.float32),
            "fxb": torch.tensor(fxb, dtype=torch.float32),
        }


class AirSimStereoDataset(_StereoSequenceDataset):
    """Sequences recorded by collect_dataset.py under one or more roots."""

    def __init__(self, roots, **kw):
        super().__init__(**kw)
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        seq_dirs = []
        for root in roots:
            root = os.path.expanduser(str(root))
            if os.path.isfile(os.path.join(root, "calib.json")):
                seq_dirs.append(root)  # a single sequence dir was passed
            else:
                seq_dirs += sorted(d for d in glob.glob(os.path.join(root, "*"))
                                   if os.path.isfile(os.path.join(d, "calib.json")))
        if not seq_dirs:
            raise FileNotFoundError(f"No sequences (calib.json) found under {roots}")

        for d in seq_dirs:
            with open(os.path.join(d, "calib.json")) as f:
                calib = json.load(f)
            poses = {}
            with open(os.path.join(d, "poses.csv")) as f:
                for row in csv.DictReader(f):
                    poses[int(row["idx"])] = world_T_optical(
                        [float(row["cam_px"]), float(row["cam_py"]), float(row["cam_pz"])],
                        [float(row["cam_qw"]), float(row["cam_qx"]),
                         float(row["cam_qy"]), float(row["cam_qz"])])
            frames = sorted(int(os.path.splitext(os.path.basename(p))[0])
                            for p in glob.glob(os.path.join(d, "depth", "*.npy")))
            frames = [i for i in frames if i in poses]
            self.sequences.append({"dir": d, "frames": frames, "calib": calib,
                                   "poses": [poses[i] for i in frames]})
        self._build_index()

    def _load_frame(self, seq, i):
        name = f"{seq['frames'][i]:06d}"
        left = cv2.imread(os.path.join(seq["dir"], "left", name + ".png"))
        right = cv2.imread(os.path.join(seq["dir"], "right", name + ".png"))
        depth = np.load(os.path.join(seq["dir"], "depth", name + ".npy"))
        return left, right, depth


def read_tartanair_depth(path):
    """TartanAir depth in metres: either a float32 .npy or the lossless
    4-channel PNG that packs the float32 bytes (V2's default download).

    The PNG decode reinterprets the raw uint8 channels as little-endian
    float32, exactly as tartanairpy's own depth_rgba_float32 does - the bytes
    round-trip through cv2 on both the write and the read side, so the
    channels must NOT be reordered to RGBA here.
    """
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"not a 4-channel packed-float32 TartanAir depth PNG: {path}")
    return np.squeeze(np.ascontiguousarray(rgba).view("<f4"), axis=-1).astype(np.float32)


class TartanAirDataset(_StereoSequenceDataset):
    """TartanAir trajectories - both the V1 and V2 layouts (https://tartanair.org/).

    V1 (tartanair_tools, 640x480):
        <Env>/<Easy|Hard>/P000/image_left/000000_left.png
                              /image_right/000000_right.png
                              /depth_left/000000_left_depth.npy
                              /pose_left.txt
    V2 (tartanairpy, 640x640):
        <Env>/Data_<easy|hard>/P000/image_lcam_front/000000.png
                                   /image_rcam_front/000000.png
                                   /depth_lcam_front/000000.npy   (or *_depth.png)
                                   /pose_lcam_front.txt

    Both versions are distortion-free pinhole with fx = fy = 320 (90 deg FOV)
    and a 0.25 m stereo baseline, so fx*baseline = 80 px*m. That is ~3.5x this
    project's own rig, i.e. TartanAir disparities are ~3.5x larger at the same
    depth. `scale` downsamples the images, which scales fx - and therefore
    disparity - linearly, so scale = target_fxb / 80 lines the two up.

    Depth is planar z in metres. Poses are "tx ty tz qx qy qz qw" in NED with
    the camera x=forward, y=right, z=down - what world_T_optical expects.
    """

    CALIB_V1 = {"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0, "baseline_m": 0.25}
    CALIB_V2 = {"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 320.0, "baseline_m": 0.25}
    RES_V1 = (640, 480)  # (W, H)
    RES_V2 = (640, 640)
    CALIB = CALIB_V1  # backwards-compatible alias

    def __init__(self, roots, camera="front", difficulty=None, envs=None,
                 scale=1.0, **kw):
        super().__init__(**kw)
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        keep_envs = set(envs) if envs else None

        traj = []
        for root in roots:
            traj += self._find_trajectories(os.path.expanduser(str(root)), camera)
        traj = sorted(set(traj))
        if not traj:
            raise FileNotFoundError(self._missing_data_message(roots, camera))

        for d, ver in traj:
            if difficulty and not self._matches_difficulty(d, difficulty):
                continue
            env = self._env_of(d)
            if keep_envs is not None and env not in keep_envs:
                continue

            if ver == 2:
                dirs = (f"image_lcam_{camera}", f"image_rcam_{camera}",
                        f"depth_lcam_{camera}")
                calib, res = dict(self.CALIB_V2), self.RES_V2
            else:
                dirs = ("image_left", "image_right", "depth_left")
                calib, res = dict(self.CALIB_V1), self.RES_V1

            lefts = sorted(glob.glob(os.path.join(d, dirs[0], "*.png")))
            rights = sorted(glob.glob(os.path.join(d, dirs[1], "*.png")))
            depths = (sorted(glob.glob(os.path.join(d, dirs[2], "*.npy")))
                      or sorted(glob.glob(os.path.join(d, dirs[2], "*.png"))))
            poses = self._read_poses(d, camera, ver)
            n = min(len(lefts), len(rights), len(depths), len(poses))
            if n == 0:
                continue

            size = self._scale_calib(calib, res, scale)
            self.sequences.append({"dir": d, "env": env, "version": ver,
                                   "frames": list(range(n)), "size": size,
                                   "files": (lefts[:n], rights[:n], depths[:n]),
                                   "calib": calib, "poses": poses[:n]})
        if not self.sequences:
            raise FileNotFoundError(
                f"{len(traj)} TartanAir trajectories found under {roots}, but none "
                f"survived the filters (difficulty={difficulty}, envs={envs}).")
        self._check_resolution()
        self._build_index()

    # -- discovery -------------------------------------------------------
    @staticmethod
    def _pose_names(camera, version):
        if version == 1:
            return ["pose_left.txt"]
        # tartanairpy writes pose_lcam_<dir>.txt; some mirrors ship pose_lcam.txt.
        return [f"pose_lcam_{camera}.txt"] + (["pose_lcam.txt"] if camera == "front" else [])

    @classmethod
    def _find_trajectories(cls, root, camera):
        """Walk `root` and return [(trajectory_dir, layout_version), ...]."""
        found = []
        v2_poses = cls._pose_names(camera, 2)
        for d, _sub, files in os.walk(root):
            if (any(p in files for p in v2_poses)
                    and os.path.isdir(os.path.join(d, f"image_lcam_{camera}"))):
                found.append((d, 2))
            elif ("pose_left.txt" in files
                  and os.path.isdir(os.path.join(d, "image_left"))):
                found.append((d, 1))
        return found

    @classmethod
    def _read_poses(cls, d, camera, version):
        path = next((os.path.join(d, n) for n in cls._pose_names(camera, version)
                     if os.path.isfile(os.path.join(d, n))), None)
        poses = []
        with open(path) as f:
            for line in f:
                v = [float(x) for x in line.split()]
                if len(v) != 7:
                    continue
                tx, ty, tz, qx, qy, qz, qw = v
                poses.append(world_T_optical([tx, ty, tz], [qw, qx, qy, qz]))
        return poses

    @staticmethod
    def _env_of(traj_dir):
        """<Env>/<difficulty>/P000 -> "<Env>" (both layouts nest the same way)."""
        return os.path.basename(os.path.dirname(os.path.dirname(traj_dir))) \
            or os.path.basename(os.path.dirname(traj_dir))

    @staticmethod
    def _matches_difficulty(traj_dir, difficulty):
        """Match "easy"/"hard" against V2's Data_easy and V1's Easy."""
        want = difficulty.lower()
        return any(part.lower() in (want, f"data_{want}")
                   for part in traj_dir.split(os.sep))

    @staticmethod
    def _scale_calib(calib, res, scale):
        """Shrink intrinsics in place for a `scale` resize; returns (w, h)."""
        W, H = res
        w, h = max(8, round(W * scale)), max(8, round(H * scale))
        sx, sy = w / W, h / H
        calib["fx"] *= sx
        calib["fy"] *= sy
        calib["cx"] = (calib["cx"] + 0.5) * sx - 0.5
        calib["cy"] = (calib["cy"] + 0.5) * sy - 0.5
        return w, h

    def _check_resolution(self):
        """Intrinsics are hardcoded per layout version, so a trajectory at an
        unexpected resolution would be silently mis-calibrated - fail loudly."""
        seq = self.sequences[0]
        img = cv2.imread(seq["files"][0][0])
        if img is None:
            raise FileNotFoundError(f"cannot read {seq['files'][0][0]}")
        expect = self.RES_V2 if seq["version"] == 2 else self.RES_V1
        if (img.shape[1], img.shape[0]) != expect:
            raise ValueError(
                f"{seq['dir']} is {img.shape[1]}x{img.shape[0]}, but TartanAir "
                f"V{seq['version']} is expected to be {expect[0]}x{expect[1]}; "
                "the hardcoded fx/fy/cx/cy would be wrong for this data.")

    # --------------------------------------------------------------------
    def _load_frame(self, seq, i):
        lefts, rights, depths = seq["files"]
        left = cv2.imread(lefts[i])
        right = cv2.imread(rights[i])
        depth = read_tartanair_depth(depths[i])
        w, h = seq["size"]
        if (depth.shape[1], depth.shape[0]) != (w, h):
            left = cv2.resize(left, (w, h), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)
            # nearest for depth: averaging across a depth edge invents surfaces
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return left, right, depth

    @staticmethod
    def _missing_data_message(roots, camera):
        return (
            f"No TartanAir trajectories found under {roots}.\n"
            f"Expected a V2 tree (<Env>/Data_easy/P000/pose_lcam_{camera}.txt) or a "
            "V1 tree (<Env>/Easy/P000/pose_left.txt).\nTo fetch V2 data:\n"
            "  pip install tartanair\n"
            "  python -c \"import tartanair as ta; ta.init('<root>'); ta.download("
            "env=['AbandonedFactory'], difficulty=['easy'], modality=['image','depth'], "
            "camera_name=['lcam_front','rcam_front'], unzip=True)\"")


def build_dataset(name, roots, **kw):
    if name == "airsim":
        return AirSimStereoDataset(roots, **kw)
    if name == "tartanair":
        return TartanAirDataset(roots, **kw)
    raise ValueError(f"unknown dataset '{name}'")
