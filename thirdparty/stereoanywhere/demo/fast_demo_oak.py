import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import tqdm
import os
import depthai as dai
from collections import deque

from torch import autocast

# Custom models
from models.stereoanywhere import StereoAnywhere as StereoAnywhere

# Monocular models - VANILLA
from models.depth_anything_v2 import get_depth_anything_v2

torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision("high")

from demo.fast_demo_utils import StereoAnywhereWrapper, dav2_rt_infer, load_engine


def getDisparityFrame(frame, cvColorMap, maxDisp=None):
    if maxDisp is None:
        maxDisp = frame.max()
    frame = np.clip(frame, 0, maxDisp)
    disp = (frame * (255.0 / maxDisp)).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cvColorMap)

    return disp


prev_disparity = None


def temporalFilter(disparity, alpha=0.15):
    global prev_disparity
    if prev_disparity is None:
        prev_disparity = disparity
    else:
        _tmp = disparity * alpha + prev_disparity * (1 - alpha)
        prev_disparity = _tmp
        disparity = _tmp
    return disparity


def deque_mean(deq):
    if len(deq) == 0:
        return 0
    return sum(deq) / len(deq)


def main():
    parser = argparse.ArgumentParser(description="StereoAnywhere OAK-D demo")

    parser.add_argument(
        "--iscale", type=float, default=1.0, help="scale factor for input images"
    )

    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="use torch.compile for optimization",
    )
    parser.add_argument(
        "--half", action="store_true", help="use half precision for inference"
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="use mixed precision for inference",
    )

    parser.add_argument(
        "--stereomodel", default="stereoanywhere", help="select stereo model"
    )
    parser.add_argument("--monomodel", default="DAv2", help="select mono model")

    parser.add_argument("--loadstereomodel", required=True, help="load stereo model")
    parser.add_argument("--loadmonomodel", required=True, help="load mono model")

    parser.add_argument(
        "--mono_width", type=int, default=518, help="Input width for the mono model"
    )
    parser.add_argument(
        "--mono_height", type=int, default=518, help="Input height for the mono model"
    )
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--n_additional_hourglass", type=int, default=0)
    parser.add_argument("--volume_channels", type=int, default=8)
    parser.add_argument("--vol_downsample", type=float, default=0)
    parser.add_argument("--vol_n_masks", type=int, default=8)
    parser.add_argument("--use_truncate_vol", action="store_true")
    parser.add_argument("--mirror_conf_th", type=float, default=0.98)
    parser.add_argument("--mirror_attenuation", type=float, default=0.9)
    parser.add_argument("--use_aggregate_stereo_vol", action="store_true")
    parser.add_argument("--use_aggregate_mono_vol", action="store_true")
    parser.add_argument("--normal_gain", type=int, default=10)
    parser.add_argument("--lrc_th", type=float, default=1.0)
    parser.add_argument(
        "--iters",
        type=int,
        default=32,
        help="Number of iterations for recurrent networks",
    )

    parser.add_argument(
        "--clip_bottom", type=int, default=0, help="Clip bottom of the image"
    )
    parser.add_argument("--clip_top", type=int, default=0, help="Clip top of the image")
    parser.add_argument(
        "--clip_left", type=int, default=0, help="Clip left of the image"
    )
    parser.add_argument(
        "--clip_right", type=int, default=0, help="Clip right of the image"
    )

    # OAK-D specific arguments
    parser.add_argument(
        "-res",
        "--resolution",
        type=str,
        default="400",
        help="Sets the resolution on mono cameras. Options: 800 | 720 | 400",
    )

    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.0,
        help="Alpha scaling parameter to increase FOV",
    )

    args = parser.parse_args()

    # Check the original code: https://docs.luxonis.com/software/depthai/examples/stereo_depth_video/
    RES_MAP = {
        "800": {
            "w": 1280,
            "h": 800,
            "res": dai.MonoCameraProperties.SensorResolution.THE_800_P,
        },
        "720": {
            "w": 1280,
            "h": 720,
            "res": dai.MonoCameraProperties.SensorResolution.THE_720_P,
        },
        "400": {
            "w": 640,
            "h": 400,
            "res": dai.MonoCameraProperties.SensorResolution.THE_400_P,
        },
    }
    if args.resolution not in RES_MAP:
        exit("Unsupported resolution!")

    resolution = RES_MAP[args.resolution]

    dtype = torch.float16 if args.half else torch.float32

    stereonet = StereoAnywhere(args)

    stereonet = nn.DataParallel(stereonet)
    pretrain_dict = torch.load(args.loadstereomodel, map_location="cpu")
    pretrain_dict = (
        pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
    )
    stereonet.load_state_dict(pretrain_dict, strict=True)
    stereonet = stereonet.module
    stereonet = stereonet.eval()

    if args.monomodel == "DAv2":
        mono_model = get_depth_anything_v2(args.loadmonomodel).eval().to(dtype)
    elif args.monomodel == "DAv2RT":
        mono_model = None
        dav2rt_engine = load_engine(args.loadmonomodel)

    wrapper = StereoAnywhereWrapper(args, stereonet, mono_model)
    wrapper = wrapper.cuda().eval().to(dtype)
    optimized_model = torch.compile(wrapper) if args.torch_compile else wrapper

    # OAK-D specific

    # Create pipeline
    device = dai.Device()
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camLeft = pipeline.create(dai.node.MonoCamera)
    camRight = pipeline.create(dai.node.MonoCamera)
    camRgb = pipeline.create(dai.node.ColorCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutDisparity = pipeline.create(dai.node.XLinkOut)
    xoutRectifLeft = pipeline.create(dai.node.XLinkOut)
    xoutRectifRight = pipeline.create(dai.node.XLinkOut)

    camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

    for monoCam in (camLeft, camRight):  # Common config
        monoCam.setResolution(resolution["res"])

    camRgb.setPreviewSize(640, 400)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(
        dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    )  # KERNEL_7x7 default
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(3)

    stereo.setAlphaScaling(args.alpha)
    config = stereo.initialConfig.get()
    config.postProcessing.brightnessFilter.minBrightness = 0
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 400
    config.postProcessing.thresholdFilter.maxRange = 15000
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.initialConfig.set(config)

    xoutRgb.setStreamName("rgb")
    xoutDisparity.setStreamName("disparity")
    xoutRectifLeft.setStreamName("rectifiedLeft")
    xoutRectifRight.setStreamName("rectifiedRight")

    camLeft.out.link(stereo.left)
    camRight.out.link(stereo.right)
    camRgb.preview.link(xoutRgb.input)

    stereo.disparity.link(xoutDisparity.input)
    stereo.rectifiedLeft.link(xoutRectifLeft.input)
    stereo.rectifiedRight.link(xoutRectifRight.input)

    streams = ["rgb", "rectifiedLeft", "rectifiedRight", "disparity"]

    # Custom color map for disparity
    cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    cvColorMap[0] = [0, 0, 0]

    cv2.namedWindow("Stereo Anywhere (OURS)", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("OAK-D SGM", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("RGB", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("LEFT", cv2.WINDOW_AUTOSIZE)

    maxDisp = deque(maxlen=10)
    maxDisp.append(stereo.initialConfig.getMaxDisparity())

    with device:
        device.startPipeline(pipeline)

        # Create a receive queue for each stream
        qList = [device.getOutputQueue(stream, 8, blocking=False) for stream in streams]

        while True:
            rect_imgs_dict = {}

            for q in qList:
                name = q.getName()
                frame = q.get().getCvFrame()

                if name == "disparity":
                    frame = frame / (2**3)
                    # print(f"Max Disparity OAK: {frame.max()}")

                    frame_copy = frame.copy()

                    # Apply clipping
                    if (
                        args.clip_bottom > 0
                        or args.clip_top > 0
                        or args.clip_left > 0
                        or args.clip_right > 0
                    ):
                        frame_copy = frame_copy[
                            args.clip_top : frame.shape[0] - args.clip_bottom,
                            args.clip_left : frame.shape[1] - args.clip_right,
                        ]

                    frame_copy = getDisparityFrame(frame_copy, cvColorMap, None)

                    cv2.imshow("OAK-D SGM", frame_copy)
                elif name == "rectifiedLeft":
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    rect_imgs_dict["rectifiedLeft"] = frame

                    frame_copy = frame.copy()
                    # Apply clipping
                    if (
                        args.clip_bottom > 0
                        or args.clip_top > 0
                        or args.clip_left > 0
                        or args.clip_right > 0
                    ):
                        frame_copy = frame_copy[
                            args.clip_top : frame.shape[0] - args.clip_bottom,
                            args.clip_left : frame.shape[1] - args.clip_right,
                        ]

                    cv2.imshow("LEFT", frame_copy)
                elif name == "rectifiedRight":
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    rect_imgs_dict["rectifiedRight"] = frame
                elif name == "rgb":
                    cv2.imshow("RGB", frame)

            if "rectifiedLeft" in rect_imgs_dict and "rectifiedRight" in rect_imgs_dict:
                left_image = rect_imgs_dict["rectifiedLeft"]
                right_image = rect_imgs_dict["rectifiedRight"]

                original_shape = left_image.shape

                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

                # Mono model inference
                if args.monomodel == "DAv2RT":
                    # start_time = time.time()
                    mono_left, mono_right = dav2_rt_infer(
                        dav2rt_engine, [left_image, right_image]
                    )
                    # mono_time = time.time() - start_time
                    # print(f"Mono inference time: {mono_time:.4f} seconds")

                    mono_left = torch.from_numpy(mono_left).unsqueeze(0).unsqueeze(0)
                    mono_right = torch.from_numpy(mono_right).unsqueeze(0).unsqueeze(0)
                    mono_depths = torch.cat([mono_left, mono_right], 0)
                    mono_depths = (mono_depths - mono_depths.min()) / (
                        mono_depths.max() - mono_depths.min()
                    )
                    mono_left = mono_depths[0].unsqueeze(0).cuda().to(dtype)
                    mono_right = mono_depths[1].unsqueeze(0).cuda().to(dtype)
                else:
                    mono_left = None
                    mono_right = None

                left_image = cv2.resize(
                    left_image,
                    (
                        round(original_shape[1] / args.iscale),
                        round(original_shape[0] / args.iscale),
                    ),
                )
                right_image = cv2.resize(
                    right_image,
                    (
                        round(original_shape[1] / args.iscale),
                        round(original_shape[0] / args.iscale),
                    ),
                )
                mono_left = (
                    F.interpolate(
                        mono_left,
                        size=(
                            round(original_shape[0] / args.iscale),
                            round(original_shape[1] / args.iscale),
                        ),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if mono_left is not None
                    else None
                )
                mono_right = (
                    F.interpolate(
                        mono_right,
                        size=(
                            round(original_shape[0] / args.iscale),
                            round(original_shape[1] / args.iscale),
                        ),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if mono_right is not None
                    else None
                )

                # Prepare inputs
                left_image = (
                    (torch.from_numpy(left_image).permute(2, 0, 1).unsqueeze(0) / 255.0)
                    .cuda()
                    .to(dtype)
                )
                right_image = (
                    (
                        torch.from_numpy(right_image).permute(2, 0, 1).unsqueeze(0)
                        / 255.0
                    )
                    .cuda()
                    .to(dtype)
                )

                inputs = (left_image, right_image, mono_left, mono_right)
                with autocast("cuda", enabled=args.mixed_precision):
                    pred_disp = optimized_model(*inputs)

                pred_disp = pred_disp.detach().squeeze().float().cpu().numpy()
                pred_disp = temporalFilter(pred_disp, alpha=0.15)
                # print(f"Max Disparity OURS: {pred_disp.max()}")
                maxDisp.append(pred_disp.max())
                pred_disp = cv2.resize(
                    pred_disp, (original_shape[1], original_shape[0])
                )

                # Apply clipping
                frame_copy = pred_disp.copy()
                if (
                    args.clip_bottom > 0
                    or args.clip_top > 0
                    or args.clip_left > 0
                    or args.clip_right > 0
                ):
                    frame_copy = frame_copy[
                        args.clip_top : pred_disp.shape[0] - args.clip_bottom,
                        args.clip_left : pred_disp.shape[1] - args.clip_right,
                    ]

                frame_copy = getDisparityFrame(frame_copy, cvColorMap, None)

                cv2.imshow("Stereo Anywhere (OURS)", frame_copy)

            else:
                print("Waiting for rectified images...")

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
