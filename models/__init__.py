from .baseline_net import FastStereoNet
from .stereo_conv_net import StereoConvNet
from .stereo_conv3d_net import StereoConv3DNet
from .temporal_net import TempoBandNet
from .mobilenet_stereo import MobileStereoNet
from .anynet_stereo import AnyStereoNet
from .common import count_parameters, upsample_disp


def build_model(name, max_disp=128, **kw):
    if name == "baseline":
        return FastStereoNet(max_disp=max_disp)
    if name == "stereoconv":
        return StereoConvNet(max_disp=max_disp)
    if name in ("stereoconv3d", "stereoconv3d_fast"):
        # Same architecture either way - the T=10 vs T=3 window length that
        # gives the two --model names their different names/costs is a
        # dataset/dataloader parameter (Section 8 / Section 9), not a model one.
        return StereoConv3DNet(max_disp=max_disp)
    if name == "temporal":
        return TempoBandNet(max_disp=max_disp)
    if name == "mobilenet":
        return MobileStereoNet(max_disp=max_disp)
    if name == "anynet":
        return AnyStereoNet(max_disp=max_disp)
    if name == "yolo":
        # Imported lazily: this is the only model that needs ultralytics, and
        # training any of the others should not require it to be installed.
        from .yolo_stereo import YoloStereoNet
        return YoloStereoNet(max_disp=max_disp,
                             scale=kw.get("yolo_scale", "n"),
                             full_res_refine=kw.get("yolo_refine", True))
    raise ValueError(f"unknown model '{name}'")
