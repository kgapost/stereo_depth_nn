from .baseline_net import FastStereoNet, FastStereoNetFxb
from .stereo_conv_net import StereoConvNet
from .stereo_conv3d_net import StereoConv3DNet
from .temporal_net import TempoBandNet
from .mobilenet_stereo import MobileStereoNet, MobileStereoNetFxb
from .anynet_stereo import AnyStereoNet, AnyStereoNetFxb
from .common import count_parameters, upsample_disp


def build_model(name, max_disp=128, **kw):
    if name == "siam2d_3dhg":
        return FastStereoNet(max_disp=max_disp)
    if name == "siam2d_3dhg_fxb":
        return FastStereoNetFxb(max_disp=max_disp)
    if name == "siam2d_2dun_fxb":
        return StereoConvNet(max_disp=max_disp)
    if name in ("c3d_3dhg_10_fxb", "c3d_3dhg_3_fxb"):
        # Same architecture either way - the T=10 vs T=3 window length that
        # gives the two --model names their different names/costs is a
        # dataset/dataloader parameter (Section 8 / Section 9), not a model one.
        return StereoConv3DNet(max_disp=max_disp)
    if name == "siam2d_egomotion_fxb":
        return TempoBandNet(max_disp=max_disp)
    if name == "mobile2d_3dhg":
        return MobileStereoNet(max_disp=max_disp)
    if name == "mobile2d_3dhg_fxb":
        return MobileStereoNetFxb(max_disp=max_disp)
    if name == "pyr2d_casc2d":
        return AnyStereoNet(max_disp=max_disp)
    if name == "pyr2d_casc2d_fxb":
        return AnyStereoNetFxb(max_disp=max_disp)
    if name == "yolo2d_3dhg":
        # Imported lazily: this is the only model that needs ultralytics, and
        # training any of the others should not require it to be installed.
        from .yolo_stereo import YoloStereoNet
        return YoloStereoNet(max_disp=max_disp,
                             scale=kw.get("yolo_scale", "n"),
                             full_res_refine=kw.get("yolo_refine", True))
    if name == "yolo2d_3dhg_fxb":
        from .yolo_stereo import YoloStereoNetFxb
        return YoloStereoNetFxb(max_disp=max_disp,
                                scale=kw.get("yolo_scale", "n"),
                                full_res_refine=kw.get("yolo_refine", True))
    raise ValueError(f"unknown model '{name}'")
