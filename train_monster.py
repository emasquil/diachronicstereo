"""Train the MonSter stereo model on DFC-style rectified pairs.

Uses Hydra for configuration and Accelerate for distributed/mixed-precision
training. Builds StereoDFC datasets (L/R images plus disparity/no-trees
targets), applies StereoAugmentor transforms, and handles diachronic pairs by
using the no-trees disparity target while synchronic pairs use the standard GT.
Logs metrics and visualizations to TensorBoard and writes checkpoints and a
final model into a timestamped run directory alongside hparams.yml.

See the @hydra.main decorator for the active config path/name and adjust dataset
roots and training hyperparameters in the YAML files.
"""

import os
from datetime import datetime
from pathlib import Path
from turtle import right

import hydra
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from datasets.augmentations import StereoAugmentor
from datasets.our_data import StereoDFC
import thirdparty


def gray_2_colormap_np(img, cmap="rainbow", max=None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img < 0] = 0
    mask_invalid = img < 1e-10
    if max is None:
        img = img / (img.max() + 1e-8)
    else:
        img = img / (max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap


def sequence_loss(
    disp_preds,
    disp_init_pred,
    disp_gt,
    valid,
    loss_gamma=0.9,
    max_disp=192,
    ignore_border=32,
):
    """Loss function defined over sequence of flow predictions"""
    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    # valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    valid = (valid >= 0.5).contiguous() & (mag < max_disp).contiguous().unsqueeze(1)
    # Ignore borders of the image where cropped things might get invalid predictions
    if ignore_border > 0:
        # top / bottom (height)
        valid[:, :, :ignore_border, :] = False
        valid[:, :, -ignore_border:, :] = False
        # left / right (width)
        valid[..., :ignore_border] = False
        valid[..., -ignore_border:] = False
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)
    disp_loss += 1.0 * F.smooth_l1_loss(
        disp_init_pred[init_valid], disp_gt[init_valid], reduction="mean"
    )

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [
            i_loss.shape,
            valid.shape,
            disp_gt.shape,
            disp_preds[i].shape,
        ]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        "train/epe": epe.mean(),
        "train/1px": (epe < 1).float().mean(),
        "train/3px": (epe < 3).float().mean(),
        "train/5px": (epe < 5).float().mean(),
    }
    return disp_loss, metrics


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""
    DPT_params = list(map(id, model.feat_decoder.parameters()))
    rest_params = filter(
        lambda x: id(x) not in DPT_params and x.requires_grad, model.parameters()
    )

    params_dict = [
        {"params": model.feat_decoder.parameters(), "lr": args.lr / 2.0},
        {"params": rest_params, "lr": args.lr},
    ]
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        [args.lr / 2.0, args.lr],
        args.total_step + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    return optimizer, scheduler


@hydra.main(
    version_base=None,
    config_path="training_configs",
    config_name="train_dfc",
)
def main(cfg):
    # Set up run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg.logdir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    logger = get_logger(__name__)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Setup TensorBoard tracker with run name
    tb_tracker = TensorBoardTracker("", logging_dir=run_dir)
    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        log_with=tb_tracker,
        project_dir=str(run_dir),
        kwargs_handlers=[kwargs],
        step_scheduler_with_optimizer=False,
    )

    def sanitize_cfg(cfg):
        out = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                for sub_k, sub_v in sanitize_cfg(v).items():
                    out[f"{k}.{sub_k}"] = sub_v
            elif isinstance(v, list):
                out[k] = str(v)
            else:
                out[k] = v
        return out

    accelerator.init_trackers(
        project_name=cfg.project_name,
    )
    with open(run_dir / "hparams.yml", "w") as f:
        yaml.safe_dump(sanitize_cfg(OmegaConf.to_container(cfg, resolve=True)), f)

    # Datasets
    augmentor = StereoAugmentor(**cfg.augmentation)
    if cfg.train_datasets[0] == "dfc":
        train_dataset = StereoDFC(
            left_dir=os.path.join(cfg.dfc.root, "L"),
            right_dir=os.path.join(cfg.dfc.root, "R"),
            disparity_dir=os.path.join(cfg.dfc.root, "disparity"),
            # Uncomment to use no-trees disparities for diachronic pairs during training, as trees in winter/summer images might be completely different
            # disparity_dir_no_trees=os.path.join(cfg.dfc.root, "disparity_no_trees"),
            disparity_dir_no_trees=os.path.join(cfg.dfc.root, "disparity"),
            train=True,
            aois_csv=cfg.dfc.train_aois_csv,
            diachronic_list_csv=os.path.join(cfg.dfc.root, "diachronic_pairs.csv"),
            syncronic_list_csv=os.path.join(cfg.dfc.root, "sinchronic_pairs.csv"),
            crop_size=cfg.image_size,
            transforms=augmentor,
        )
    else:
        raise ValueError(f"Unknown dataset {cfg.train_datasets[0]}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size // cfg.num_gpu,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    if cfg.train_datasets[0] == "dfc":
        val_dataset = StereoDFC(
            left_dir=os.path.join(cfg.dfc.root, "L"),
            right_dir=os.path.join(cfg.dfc.root, "R"),
            disparity_dir=os.path.join(cfg.dfc.root, "disparity"),
            # disparity_dir_no_trees=os.path.join(cfg.dfc.root, "disparity_no_trees"),
            disparity_dir_no_trees=os.path.join(cfg.dfc.root, "disparity"),
            train=False,
            aois_csv=cfg.dfc.val_aois_csv,
            diachronic_list_csv=os.path.join(cfg.dfc.root, "diachronic_pairs.csv"),
            syncronic_list_csv=os.path.join(cfg.dfc.root, "sinchronic_pairs.csv"),
            crop_size=cfg.image_size,
        )
    else:
        raise ValueError(f"Unknown dataset {cfg.train_datasets[0]}")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=8
    )

    # Model
    if cfg.restore_ckpt is not None:
        assert cfg.restore_ckpt.endswith(".pth")
        assert os.path.exists(cfg.restore_ckpt)
        print(f"Loading checkpoint from {cfg.restore_ckpt}")

    model = thirdparty.build_monster(
        monster_ckpt=cfg.restore_ckpt,
        depth_anything_v2_path=cfg.depth_anything_v2_path,
        device="cpu",
        args=cfg,
        eval_only=False,
    )

    if cfg.restore_ckpt is not None:
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")

    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(
        train_loader, model, optimizer, lr_scheduler, val_loader
    )
    model.to(accelerator.device)

    total_step = 0
    should_keep_training = True
    while should_keep_training:
        model.train()
        getattr(model, "module", model).freeze_bn()
        for data in tqdm(
            train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process
        ):
            # Scale inputs
            left, right = data["left"] * 255.0, data["right"] * 255.0

            # GT and masks from the batch
            disp_gt = data["disparity"]
            valid = data.get("valid", (disp_gt > 0).float()).to(disp_gt.device)
            disp_no_trees_gt = data.get("disparity_no_trees", disp_gt)
            valid_no_trees = data.get(
                "valid_no_trees", (disp_no_trees_gt > 0).float()
            ).to(disp_no_trees_gt.device)

            # Diachronic mask; default to all False
            diachronic_mask = data.get(
                "diachronic_pair",
                torch.zeros(left.shape[0], device=left.device, dtype=torch.uint8),
            ).to(torch.bool)

            # Forward pass
            with accelerator.autocast():
                disp_init_pred, disp_preds, depth_mono = model(
                    left, right, iters=cfg.train_iters
                )

            # Compute diachronic/synchronic indices
            idx_dia = torch.nonzero(diachronic_mask, as_tuple=False).squeeze(-1)
            idx_syn = torch.nonzero(~diachronic_mask, as_tuple=False).squeeze(-1)

            B = left.shape[0]
            loss = 0.0
            metrics = {}

            # ---- 1) Diachronic pairs => use disp_no_trees_gt ----
            if idx_dia.numel() > 0:
                loss_dia, metrics_dia = sequence_loss(
                    [p[idx_dia] for p in disp_preds],  # views, no copy
                    disp_init_pred[idx_dia],  # view
                    disp_no_trees_gt[idx_dia],  # view
                    valid_no_trees[idx_dia],  # view
                    max_disp=cfg.max_disp,
                )
                # Weight by fraction of the batch (keeps overall scale stable)
                loss = loss + loss_dia * (idx_dia.numel() / B)
                # Keep metrics with suffix “_dia”
                for k, v in metrics_dia.items():
                    metrics[f"{k}_dia"] = v

            # ---- 2) Non-diachronic pairs ----
            if idx_syn.numel() > 0:
                # Use normal GT on non-diachronic
                loss_syn, metrics_syn = sequence_loss(
                    [p[idx_syn] for p in disp_preds],
                    disp_init_pred[idx_syn],
                    disp_gt[idx_syn],
                    valid[idx_syn],
                    max_disp=cfg.max_disp,
                )
                loss = loss + loss_syn * (idx_syn.numel() / B)
                for k, v in metrics_syn.items():
                    metrics[f"{k}_syn_gt"] = v

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_step += 1
            accelerator.log(
                {
                    "train/loss": accelerator.reduce(loss.detach(), "mean"),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=total_step,
            )
            accelerator.log(accelerator.reduce(metrics, "mean"), step=total_step)

            if total_step % 20 == 0 and accelerator.is_main_process:
                # Apply consistent colormap scaling
                # Get max disparity across GT and prediction
                disp_pred = disp_preds[-1][0].squeeze()
                disp_gt_img = disp_gt[0].squeeze()
                shared_max = torch.max(disp_pred.max(), disp_gt_img.max()).item()
                disp_pred_np = gray_2_colormap_np(disp_pred, max=shared_max)
                disp_gt_np = gray_2_colormap_np(disp_gt_img, max=shared_max)
                depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze())

                tb_tracker.writer.add_image(
                    "disp_pred", to_tensor(disp_pred_np), total_step
                )
                tb_tracker.writer.add_image(
                    "disp_gt", to_tensor(disp_gt_np), total_step
                )
                tb_tracker.writer.add_image(
                    "depth_mono", to_tensor(depth_mono_np), total_step
                )
                tb_tracker.writer.flush()

            if total_step % cfg.save_frequency == 0 and accelerator.is_main_process:
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    run_dir / f"{total_step}.pth",
                )

            if total_step % cfg.val_frequency == 0:
                model.eval()
                elem_num, total_epe, total_out = 0, 0, 0
                for val_data in tqdm(
                    val_loader,
                    dynamic_ncols=True,
                    disable=not accelerator.is_main_process,
                ):
                    left, right = val_data["left"] * 255.0, val_data["right"] * 255.0
                    disp_gt = val_data["disparity"]
                    valid = val_data.get("valid", (disp_gt > 0).float())

                    padder = thirdparty.MonsterInputPadder(left.shape, divis_by=32)
                    left, right = padder.pad(left, right)
                    with torch.no_grad():
                        disp_pred = model(
                            left, right, iters=cfg.valid_iters, test_mode=True
                        )
                    disp_pred = padder.unpad(disp_pred)

                    epe = torch.abs(disp_pred - disp_gt)
                    out = (epe > 1.0).float()

                    # latest change without accelerate
                    # epe, out = accelerator.gather_for_metrics(
                    #     (epe[valid >= 0.5].mean(), out[valid >= 0.5].mean())
                    # )
                    # elem_num += epe.shape[0]
                    epe, out = accelerator.gather_for_metrics(
                        (
                            epe[valid >= 0.5].mean().unsqueeze(0),
                            out[valid >= 0.5].mean().unsqueeze(0),
                        )
                    )
                    elem_num += epe.numel()
                    # latest change without accelerate

                    total_epe += epe.sum()
                    total_out += out.sum()

                    # Log validation images for first val batch
                    if total_step % 20 == 0 and accelerator.is_main_process:
                        shared_max = torch.max(disp_pred.max(), disp_gt.max()).item()
                        val_disp_pred_np = gray_2_colormap_np(
                            disp_pred.squeeze(), max=shared_max
                        )
                        val_disp_gt_np = gray_2_colormap_np(
                            disp_gt.squeeze(), max=shared_max
                        )
                        tb_tracker.writer.add_image(
                            "val/disp_pred", to_tensor(val_disp_pred_np), total_step
                        )
                        tb_tracker.writer.add_image(
                            "val/disp_gt", to_tensor(val_disp_gt_np), total_step
                        )
                        tb_tracker.writer.flush()

                accelerator.log(
                    {
                        "val/epe": total_epe / elem_num,
                        "val/d1": 100 * total_out / elem_num,
                    },
                    total_step,
                )
                model.train()
                getattr(model, "module", model).freeze_bn()

            if total_step >= cfg.total_step:
                should_keep_training = False
                break

    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), run_dir / "final.pth")
    accelerator.end_training()


if __name__ == "__main__":
    main()
