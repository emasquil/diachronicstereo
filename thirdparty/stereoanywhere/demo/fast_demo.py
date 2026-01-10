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
from torch import autocast

# add the parent directory to the system path
import sys

sys.path.append("..")

# Custom models
from models.stereoanywhere import StereoAnywhere as StereoAnywhere

# Monocular models - VANILLA
from models.depth_anything_v2 import get_depth_anything_v2

from demo.fast_demo_utils import StereoAnywhereWrapper, dav2_rt_infer, load_engine

torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision("high")


def main():
    parser = argparse.ArgumentParser(description="StereoAnywhere Fast Inference")

    parser.add_argument("--left", nargs="+", required=True, help="left image path(s)")
    parser.add_argument("--right", nargs="+", required=True, help="right image path(s)")

    parser.add_argument(
        "--iscale", type=float, default=1.0, help="scale factor for input images"
    )
    parser.add_argument(
        "--outdir",
        default=None,
        type=str,
        help="output directory. If not specified (None), will be saved in the same directory as input images with .npy extension",
    )
    parser.add_argument(
        "--display_qualitatives",
        action="store_true",
        help="display qualitative results",
    )
    parser.add_argument(
        "--save_qualitatives", action="store_true", help="save qualitative results"
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

    args = parser.parse_args()

    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)

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

    if args.display_qualitatives:
        cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)

    for _left, _right in tqdm.tqdm(
        zip(args.left, args.right),
        desc="Processing stereo images",
        total=len(args.left),
    ):

        # Load images
        left_image = cv2.imread(_left)
        right_image = cv2.imread(_right)

        # check if images are grayscale and convert to BGR
        if len(left_image.shape) == 2:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
        if len(right_image.shape) == 2:
            right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

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

            if args.save_qualitatives or args.display_qualitatives:
                mono_left_jet = (mono_left - mono_left.min()) / (
                    mono_left.max() - mono_left.min()
                )
                mono_left_jet = (mono_left_jet * 255).astype(np.uint8)
                mono_left_jet = cv2.applyColorMap(mono_left_jet, cv2.COLORMAP_JET)

                if args.save_qualitatives:
                    _output = (
                        f"{os.path.splitext(_left)[0]}_mono_left.png"
                        if args.outdir is None
                        else os.path.join(
                            args.outdir,
                            os.path.splitext(os.path.basename(_left))[0]
                            + "_mono_left.png",
                        )
                    )
                    cv2.imwrite(_output, mono_left_jet)

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

        # Save the output
        _output = (
            f"{os.path.splitext(_left)[0]}.npy"
            if args.outdir is None
            else os.path.join(
                args.outdir, os.path.splitext(os.path.basename(_left))[0] + ".npy"
            )
        )
        np.save(_output, pred_disp)

        if args.save_qualitatives or args.display_qualitatives:
            pred_disp = (pred_disp - pred_disp.min()) / (
                pred_disp.max() - pred_disp.min()
            )

            pred_disp = (pred_disp * 255).astype(np.uint8)
            pred_disp = cv2.applyColorMap(pred_disp, cv2.COLORMAP_JET)

            if args.save_qualitatives:
                _output = (
                    f"{os.path.splitext(_left)[0]}_disp_jet.png"
                    if args.outdir is None
                    else os.path.join(
                        args.outdir,
                        os.path.splitext(os.path.basename(_left))[0] + "_disp_jet.png",
                    )
                )
                cv2.imwrite(_output, pred_disp)

            if args.display_qualitatives:
                cv2.imshow("Disparity", pred_disp)
                cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
