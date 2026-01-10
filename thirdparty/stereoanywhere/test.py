from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

from torch import autocast

# Custom models

from models.stereoanywhere import StereoAnywhere as StereoAnywhere

# Monocular models
from models.depth_anything_v2 import get_depth_anything_v2

import cmapy

from losses import *
import cv2

from utils import color_error_image_kitti, guided_visualize

from dataloaders import fetch_dataloader
import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser(description="StereoAnywhere")

parser.add_argument("--maxdisp", type=int, default=192, help="maxium disparity")

parser.add_argument("--stereomodel", default="stereoanywhere", help="select model")

parser.add_argument("--datapath", default="dataset/oak_dataset/", help="datapath")
parser.add_argument("--dataset", default="middlebury", help="dataset type")

parser.add_argument("--outdir", default=None)

parser.add_argument("--loadstereomodel", required=True, help="load stereo model")

parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)

parser.add_argument("--iscale", type=float, default=1.0, help="Downsampling factor")
parser.add_argument("--oscale", type=float, default=1.0, help="Downsampling factor")

parser.add_argument("--tries", type=int, default=1)

parser.add_argument("--csv_path", default=None)

parser.add_argument("--mixed_precision", action="store_true")

parser.add_argument("--numworkers", type=int, default=1)
parser.add_argument("--verbose", action="store_true")
parser.add_argument(
    "--errormetric",
    default="bad 3.0",
    choices=["bad 1.0", "bad 2.0", "bad 3.0", "bad 4.0", "avgerr", "rms"],
    help="metric for errormap text",
)
parser.add_argument("--dilation", type=int, default=1)
parser.add_argument("--normalize", action="store_true")
parser.add_argument(
    "--valsize", default=0, type=int, help="Number of frames to evaluate"
)

parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 1)"
)
parser.add_argument("--vanilla", action="store_true")

parser.add_argument("--monomodel", default="DAv2", help="select model")

parser.add_argument("--loadmonomodel", default=None, help="load model")

parser.add_argument("--overfit", action="store_true", default=False, help="overfit")

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
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

assert args.iscale > 0
assert args.oscale > 0

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

autocast_device = str(device).split(":")[0]

stereonet = None

if args.stereomodel == "stereoanywhere":
    stereonet = StereoAnywhere(args)
elif args.stereomodel in ["skip_pred"]:
    stereonet = None
else:
    print("no model")
    exit()

if args.monomodel == "DAv2":
    mono_model = get_depth_anything_v2(args.loadmonomodel)
    mono_model = mono_model.to(device)
    mono_model.eval()
elif args.monomodel == "none":
    mono_model = None

if stereonet is not None:
    if args.stereomodel not in ["unimatch", "crestereo", "croco", "hitnet", "nmrf"]:
        stereonet = nn.DataParallel(stereonet)

    if args.cuda:
        stereonet.cuda()

    print("Load pretrained stereo model")

    if args.stereomodel in ["stereoanywhere"]:
        pretrain_dict = torch.load(args.loadstereomodel, map_location=device)
        pretrain_dict = (
            pretrain_dict["state_dict"]
            if "state_dict" in pretrain_dict
            else pretrain_dict
        )
        stereonet.load_state_dict(pretrain_dict, strict=True)
    else:
        print("no model")
        exit()

    if not args.cuda and args.stereomodel not in [
        "unimatch",
        "crestereo",
        "croco",
        "hitnet",
        "nmrf",
    ]:
        stereonet = stereonet.module.cpu()


@torch.no_grad()
def run(data):

    if "maskocc" not in data:
        data["maskocc"] = torch.zeros_like(data["gt"])

    if stereonet is not None:
        stereonet.eval()

    if args.iscale != 1:
        data["im2"] = F.interpolate(data["im2"], scale_factor=1.0 / args.iscale)
        data["im3"] = F.interpolate(data["im3"], scale_factor=1.0 / args.iscale)

    if args.oscale != 1:
        data["gt"] = (
            F.interpolate(data["gt"], scale_factor=1.0 / args.oscale, mode="nearest")
            / args.oscale
        )
        data["validgt"] = F.interpolate(
            data["validgt"], scale_factor=1.0 / args.oscale, mode="nearest"
        )
        data["maskocc"] = F.interpolate(
            data["maskocc"], scale_factor=1.0 / args.oscale, mode="nearest"
        )

    if args.cuda:
        data["im2"], data["im3"] = data["im2"].to(device), data["im3"].to(device)

    # Check if gt has points
    if data["gt"].max() == 0:
        results = guided_metrics(
            torch.zeros_like(data["gt"]).detach().cpu().numpy(),
            data["gt"].detach().cpu().numpy(),
            data["validgt"].detach().cpu().numpy(),
            data["maskocc"].detach().cpu().numpy(),
        )
        results["disp"] = torch.ones_like(data["gt"]).squeeze(1)
        data["im2_mono"] = torch.zeros_like(data["im2"])[:, 0:1]
        data["im3_mono"] = torch.zeros_like(data["im3"])[:, 0:1]
        return results

    with autocast(autocast_device, enabled=args.mixed_precision):
        if args.monomodel == "DAv2":
            # Assume batch size = 1 only for testing
            _input_size_width_dict = {
                "kitti2012": 1372,
                "kitti2015": 1372,
                "eth3d": 518,
                "middlebury": 518 * 2,
                "middlebury2021": 1372,
                "booster": 518 * 2,
                "layeredflow": 952,
            }
            _input_size_width = (
                _input_size_width_dict[args.dataset]
                if args.dataset in _input_size_width_dict
                else 518
            )
            _input_size_height_dict = {
                "kitti2012": 518,
                "kitti2015": 518,
                "eth3d": 518,
                "middlebury": 518 * 2,
                "middlebury2021": 770,
                "booster": 756,
                "layeredflow": 532,
            }
            _input_size_height = (
                _input_size_height_dict[args.dataset]
                if args.dataset in _input_size_height_dict
                else 518
            )
            mono_depths = mono_model.infer_image(
                torch.cat([data["im2"], data["im3"]], 0),
                input_size_width=_input_size_width,
                input_size_height=_input_size_height,
            )
            # Normalize depth between 0 and 1
            mono_depths = (mono_depths - mono_depths.min()) / (
                mono_depths.max() - mono_depths.min()
            )
            data["im2_mono"], data["im3_mono"] = mono_depths[0].unsqueeze(
                0
            ), mono_depths[1].unsqueeze(0)
        elif args.monomodel == "none":
            data["im2_mono"] = torch.zeros_like(data["im2"])[:, 0:1]
            data["im3_mono"] = torch.zeros_like(data["im3"])[:, 0:1]

    ht, wt = data["im2"].shape[-2], data["im2"].shape[-1]

    pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
    pad_wd = (((wt // 32) + 1) * 32 - wt) % 32

    _pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
    data["im2"] = F.pad(data["im2"], _pad, mode="replicate")
    data["im3"] = F.pad(data["im3"], _pad, mode="replicate")
    data["im2_mono"] = F.pad(data["im2_mono"], _pad, mode="replicate")
    data["im3_mono"] = F.pad(data["im3_mono"], _pad, mode="replicate")

    with autocast(autocast_device, enabled=args.mixed_precision):

        if args.stereomodel in ["stereoanywhere"]:
            pred_disps, _ = stereonet(
                data["im2"],
                data["im3"],
                data["im2_mono"],
                data["im3_mono"],
                test_mode=True,
                iters=args.iters,
            )
        elif args.stereomodel == "skip_pred":
            pred_disps = torch.zeros_like(data["im2"])
        else:
            pred_disps = stereonet(data["im2"], data["im3"])

    if args.stereomodel in ["stereoanywhere"]:
        pred_disp = -pred_disps.squeeze(1)
    elif args.stereomodel == "skip_pred":
        pred_disp = pred_disps.squeeze(1)

    hd, wd = pred_disp.shape[-2:]

    c = [_pad[2], hd - _pad[3], _pad[0], wd - _pad[1]]
    pred_disp = pred_disp[..., c[0] : c[1], c[2] : c[3]]

    data["im2"] = data["im2"][..., c[0] : c[1], c[2] : c[3]]
    data["im3"] = data["im3"][..., c[0] : c[1], c[2] : c[3]]
    data["im2_mono"] = data["im2_mono"][..., c[0] : c[1], c[2] : c[3]]
    data["im3_mono"] = data["im3_mono"][..., c[0] : c[1], c[2] : c[3]]

    if args.iscale != 1 and args.iscale / args.oscale != 1:
        pred_disp = (
            F.interpolate(
                pred_disp.unsqueeze(0),
                (data["gt"].shape[-2], data["gt"].shape[-1]),
                mode="nearest",
            ).squeeze(0)
            * args.iscale
            / args.oscale
        )

    result = {}

    if "gt" in data:
        result = guided_metrics(
            pred_disp.detach().cpu().numpy(),
            data["gt"].detach().cpu().numpy(),
            data["validgt"].detach().cpu().numpy(),
            data["maskocc"].detach().cpu().numpy(),
        )

    result["disp"] = pred_disp

    return result


def write_csv_header(file, args, metrics):
    header = "DATASET,DATAPATH,MONOSTEREOMODEL,MONOMODEL_PATH,STEREOMODEL,STEREOMODEL_PATH,TRIES,ISCALE,MAXDISP,NORMALIZE,"
    keys = list(metrics.keys())
    for k in keys[:-1]:
        header += f"{k.upper()},"
    header += f"{keys[-1].upper()}\n"

    file.write(header)


def write_csv_row(file, args, metrics):
    parameters = f"{args.dataset},{args.datapath},{args.monomodel},{args.loadmonomodel},{args.stereomodel},{args.loadstereomodel},{args.tries},{args.iscale},{args.maxdisp},{args.normalize},"
    keys = list(metrics.keys())
    for k in keys[:-1]:
        if "bad" not in k:
            parameters += f"{metrics[k]:.2f},"
        else:
            parameters += f"{metrics[k]*100:.2f},"

    if "bad" not in keys[-1]:
        parameters += f"{metrics[keys[-1]]:.2f}\n"
    else:
        parameters += f"{metrics[keys[-1]]*100:.2f}\n"

    file.write(parameters)


def main():
    args.test = True
    args.batch_size = 1
    demo_loader = fetch_dataloader(args)

    if args.outdir is not None and not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    ## demo ##
    acc_list = []

    for asd in range(args.tries):

        acc = {}
        pbar = tqdm.tqdm(total=len(demo_loader))
        val_len = (
            min(len(demo_loader), args.valsize)
            if args.valsize > 0
            else len(demo_loader)
        )
        for batch_idx, datablob in enumerate(demo_loader):
            if batch_idx >= val_len:
                break

            result = run(datablob)

            if args.outdir is not None and asd == 0:
                for dirname in [
                    "dmap",
                    "left",
                    "right",
                    "gt",
                    "maemap",
                    "metricmap",
                    "mono_left",
                    "mono_right",
                    "raw",
                ]:
                    if not os.path.exists(os.path.join(args.outdir, dirname)):
                        os.mkdir(os.path.join(args.outdir, dirname))

                max_val = torch.where(
                    torch.isinf(datablob["gt"][0]), -float("inf"), datablob["gt"][0]
                ).max()
                max_val = result["disp"][0].max().cpu() if max_val == 0 else max_val

                myleft = cv2.cvtColor(
                    (255 * datablob["im2"])
                    .squeeze()
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
                cv2.imwrite(
                    os.path.join(args.outdir, "left", "%s.png" % (batch_idx)), myleft
                )

                myright = cv2.cvtColor(
                    (255 * datablob["im3"])
                    .squeeze()
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
                cv2.imwrite(
                    os.path.join(args.outdir, "right", "%s.png" % (batch_idx)), myright
                )

                mygt = cv2.applyColorMap(
                    (
                        (torch.clamp(datablob["gt"][0, 0], 0, max_val) / max_val * 255)
                        .detach()
                        .cpu()
                        .numpy()
                    ).astype(np.uint8),
                    cmapy.cmap("Spectral_r"),
                )
                maemap = color_error_image_kitti(
                    torch.abs(datablob["gt"].cpu() - result["disp"].cpu())
                    .squeeze()
                    .numpy(),
                    scale=1,
                    mask=datablob["gt"].cpu() > 0,
                    dilation=args.dilation,
                )
                metricmap = guided_visualize(
                    result["disp"].squeeze().cpu().numpy(),
                    datablob["gt"].squeeze().cpu().numpy(),
                    datablob["gt"].squeeze().cpu().numpy() > 0,
                    dilation=args.dilation,
                )[args.errormetric]

                if args.dilation > 0:
                    kernel = np.ones((args.dilation, args.dilation))
                    mygt = cv2.dilate(mygt, kernel)
                    maemap = cv2.dilate(maemap, kernel)
                    metricmap = cv2.dilate(metricmap, kernel)

                cv2.imwrite(
                    os.path.join(args.outdir, "gt", "%s.png" % (batch_idx)), mygt
                )
                cv2.imwrite(
                    os.path.join(args.outdir, "maemap", "%s.png" % (batch_idx)), maemap
                )
                cv2.imwrite(
                    os.path.join(args.outdir, "metricmap", "%s.png" % (batch_idx)),
                    metricmap,
                )

                mydmap = cv2.applyColorMap(
                    (
                        (
                            (torch.clamp(result["disp"][0], 0, max_val) / max_val)
                            ** (1.0)
                            * 255
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    ).astype(np.uint8),
                    cmapy.cmap("Spectral_r"),
                )
                cv2.imwrite(
                    os.path.join(args.outdir, "dmap", "%s.png" % (batch_idx)), mydmap
                )
                cv2.imwrite(
                    os.path.join(args.outdir, "raw", f"{batch_idx:06}_10.png"),
                    (256.0 * result["disp"][0])
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint16),
                )

                mono_left = cv2.applyColorMap(
                    (
                        ((datablob["im2_mono"]) ** (1.0) * 255)
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                    ).astype(np.uint8),
                    cmapy.cmap("Spectral_r"),
                )
                mono_right = cv2.applyColorMap(
                    (
                        (datablob["im3_mono"] * 255).squeeze().detach().cpu().numpy()
                    ).astype(np.uint8),
                    cmapy.cmap("Spectral_r"),
                )

                cv2.imwrite(
                    os.path.join(args.outdir, "mono_left", "%s.png" % (batch_idx)),
                    mono_left,
                )
                cv2.imwrite(
                    os.path.join(args.outdir, "mono_right", "%s.png" % (batch_idx)),
                    mono_right,
                )

            for k in result:
                if k != "disp" and k != "errormap":
                    if k not in acc:
                        acc[k] = []
                    acc[k].append(result[k])

                    if args.verbose:
                        print(f"{batch_idx}) {k}: {result[k]}")

            pbar.update(1)
        pbar.close()

        acc_list.append(acc)

    acc_mean = {}
    acc_std = {}

    for acc in acc_list:
        for k in acc:
            if k not in acc_mean:
                acc_mean[k] = []
            if k not in acc_std:
                acc_std[k] = []

            acc_mean[k].append(np.nanmean(np.array(acc[k])))
            acc_std[k].append(np.nanmean(np.array(acc[k])))

    for k in acc_mean:
        acc_mean[k] = np.nanmean(acc_mean[k])
        acc_std[k] = np.nanstd(acc_std[k])

    print("MEAN Metrics:")

    metrs = ""
    for k in acc_mean:
        metrs += f" {k.upper()} &"
    print(metrs)

    metrs = ""
    for k in acc_mean:
        if "bad" not in k:
            metrs += f" {acc_mean[k]:.2f} &"
        else:
            metrs += f" {acc_mean[k]*100:.2f} &"

    print(metrs)

    print("STD Metrics:")

    metrs = ""
    for k in acc_std:
        if "bad" not in k:
            metrs += f" {acc_std[k]:.2f} &"
        else:
            metrs += f" {acc_std[k]*100:.2f} &"

    print(metrs)

    if args.csv_path is not None:
        if os.path.exists(args.csv_path):
            csv_file = open(args.csv_path, "a")
        else:
            csv_file = open(args.csv_path, "w")
            write_csv_header(csv_file, args, acc_mean)

        write_csv_row(csv_file, args, acc_mean)

        csv_file.close()


if __name__ == "__main__":
    main()
