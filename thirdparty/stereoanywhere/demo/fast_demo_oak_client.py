import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tqdm
import os
import depthai as dai
from collections import deque
import json
import tarfile
import io
import requests

torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision("high")


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
    parser = argparse.ArgumentParser(description="StereoAnywhere OAK-D demo client")

    parser.add_argument(
        "--ip", type=str, default="0.0.0.0", help="ip to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="port to run the server on"
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

    parser.add_argument(
        "--temporal_alpha", type=float, default=0.15, help="Alpha for temporal filter"
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

                files = {
                    "leftimage": cv2.imencode(
                        ".jpg", left_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                    )[1].tobytes(),
                    "rightimage": cv2.imencode(
                        ".jpg", right_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                    )[1].tobytes(),
                }

                # Send POST request with images and config
                response = requests.post(
                    f"http://{args.ip}:{args.port}/infer", files=files
                )

                # Save the result image
                if response.status_code == 200:
                    # decode tarfile
                    _tar = response.content
                    with tarfile.open(fileobj=io.BytesIO(_tar), mode="r") as tar:
                        pred_disp_byte = tar.extractfile("pred_disp.png").read()

                    pred_disp = (
                        cv2.imdecode(
                            np.frombuffer(pred_disp_byte, np.uint8), cv2.IMREAD_ANYDEPTH
                        )
                        / 256.0
                    )
                else:
                    continue

                pred_disp = temporalFilter(pred_disp, alpha=args.temporal_alpha)
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
