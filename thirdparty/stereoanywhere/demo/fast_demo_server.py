import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
import argparse
import random
import json
import os
import tarfile
from torch import autocast

# add the parent directory to the system path
import sys

sys.path.append("..")

# Custom models
from models.stereoanywhere import StereoAnywhere as StereoAnywhere

# Monocular models - VANILLA
from models.depth_anything_v2 import get_depth_anything_v2

from demo.fast_demo_utils import StereoAnywhereWrapper, dav2_rt_infer, load_engine

import tensorrt as trt
import pycuda.driver as cuda

# Use the primary context for PyCUDA instead of autoinit
# import pycuda.autoinit
cuda.init()
pycuda_ctx = cuda.Device(0).retain_primary_context()


def compress_image_lossy(image):
    """Compress the image using OpenCV and return the compressed bytes."""
    success, encoded_image = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    )
    if success:
        return encoded_image.tobytes()
    else:
        raise ValueError("Image compression failed")


def compress_image_lossless(image):
    """Compress the image using OpenCV and return the compressed bytes."""
    success, encoded_image = cv2.imencode(".png", image)
    if success:
        return encoded_image.tobytes()
    else:
        raise ValueError("Image compression failed")


app = Flask(__name__)

parser = argparse.ArgumentParser(description="StereoAnywhere Fast Inference Server")
parser.add_argument("--ip", type=str, default="0.0.0.0", help="ip to run the server on")
parser.add_argument("--port", type=int, default=5000, help="port to run the server on")

parser.add_argument(
    "--iscale", type=float, default=1.0, help="scale factor for input images"
)

parser.add_argument(
    "--torch_compile", action="store_true", help="use torch.compile for optimization"
)
parser.add_argument(
    "--half", action="store_true", help="use half precision for inference"
)
parser.add_argument(
    "--mixed_precision", action="store_true", help="use mixed precision for inference"
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
    "--iters", type=int, default=32, help="Number of iterations for recurrent networks"
)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

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


@app.route("/infer", methods=["POST"])
def infer():
    try:
        # Get images from the request
        leftimage_bytes = request.files["leftimage"].read()
        rightimage_bytes = request.files["rightimage"].read()

        leftimg_array = np.frombuffer(leftimage_bytes, np.uint8)
        rightimg_array = np.frombuffer(rightimage_bytes, np.uint8)
        left_image = cv2.imdecode(leftimg_array, cv2.IMREAD_COLOR)
        right_image = cv2.imdecode(rightimg_array, cv2.IMREAD_COLOR)

        original_shape = left_image.shape
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        # Perform inference

        if args.monomodel == "DAv2RT":
            pycuda_ctx.push()  # Use the primary context
            mono_left, mono_right = dav2_rt_infer(
                dav2rt_engine, [left_image, right_image]
            )
            pycuda_ctx.pop()  # Pop the primary context
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
            (torch.from_numpy(right_image).permute(2, 0, 1).unsqueeze(0) / 255.0)
            .cuda()
            .to(dtype)
        )

        inputs = (left_image, right_image, mono_left, mono_right)
        with autocast("cuda", enabled=args.mixed_precision):
            pred_disp = optimized_model(*inputs)

        pred_disp = pred_disp.detach().squeeze().float().cpu().numpy()
        pred_disp = cv2.resize(pred_disp, (original_shape[1], original_shape[0]))

        # Compress the image
        pred_disp = compress_image_lossless((256.0 * pred_disp).astype(np.uint16))

        # Create a tar file in memory
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            tarinfo = tarfile.TarInfo("pred_disp.png")
            tarinfo.size = len(pred_disp)
            tar.addfile(tarinfo, io.BytesIO(pred_disp))

        # Return the compressed tar file
        tar_buffer.seek(0)
        return send_file(
            tar_buffer,
            mimetype="application/x-tar",
            as_attachment=True,
            download_name="inference.tar",
        )

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400


app.run(host=args.ip, port=args.port)
