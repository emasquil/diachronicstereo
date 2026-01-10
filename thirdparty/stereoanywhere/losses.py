import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import numpy as np
import cv2


def SSIM(x: torch.Tensor, y: torch.Tensor, md=1):
    assert x.size() == y.size(), f"xsize: {x.size()}, ysize: {y.size()}"

    patch_size = 2 * md + 1
    C1 = 0.01**2
    C2 = 0.03**2
    refl = nn.ReflectionPad2d(md)

    x = refl(x)
    y = refl(y)
    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist


def CSIM(x: torch.Tensor, y: torch.Tensor, md=1):
    assert x.size() == y.size(), f"xsize: {x.size()}, ysize: {y.size()}"

    patch_size = 2 * md + 1
    K1 = 0.25
    K2 = 1
    refl = nn.ReflectionPad2d(md)

    x = refl(x)
    y = refl(y)
    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    CSIM_K1 = mu_x_sq + mu_y_sq - 2 * mu_x_mu_y
    CSIM_K2 = sigma_x + sigma_y - 2 * sigma_xy

    CSIM = K1 * CSIM_K1 + K2 * CSIM_K2

    dist = torch.clamp((1 - CSIM) / 2, 0, 1)

    return dist


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad(disp, image, alpha, order=1):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(disp)
    if order == 2:
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        dx, dy = dx2, dy2

    loss_x = weights_x[:, :, :, 1:] * dx[:, :, :, 1:].abs()
    loss_y = weights_y[:, :, 1:, :] * dy[:, :, 1:, :].abs()

    return loss_x.mean() / 2.0 + loss_y.mean() / 2.0


def loss_smooth(disp, im1_scaled):
    #    if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
    #    func_smooth = smooth_grad_2nd
    #    else:
    func_smooth = smooth_grad
    loss = []
    loss += [func_smooth(disp, im1_scaled, 1, order=1)]
    return sum([l.mean() for l in loss])


def lr_mask(disp12, disp21):
    disp21o = torch.from_numpy(
        np.ascontiguousarray(disp21.detach().cpu().numpy()[:, :, :, ::-1])
    ).cuda()
    disp12_recons = disp_warp(disp21o, disp12)
    occ12 = (disp12 - disp12_recons).abs().detach() < 3
    return occ12.astype(np.float32)


def disp_warp(x, disp, r2l=False, pad="border", mode="bilinear"):
    B, _, H, W = x.size()
    offset = -1
    if r2l:
        offset = 1

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW
    v_grid = norm_grid(
        base_grid + torch.cat((offset * disp, torch.zeros_like(disp)), 1)
    )  # BHW2
    x_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return x_recons


def loss_photometric(im1_scaled, im1_recons):
    loss = []

    loss += [0.15 * (im1_scaled - im1_recons).abs().mean(1, True)]
    loss += [0.85 * SSIM(im1_recons, im1_scaled).mean(1, True)]
    return sum([l for l in loss])


def self_supervised_loss(disp12, im1, im2, r2l=False):
    """Loss function defined over sequence of flow predictions"""

    im1_recons = disp_warp(im2, disp12, r2l)
    loss_warp, _ = torch.min(
        torch.cat(
            (loss_photometric(im1, im1_recons), loss_photometric(im2, im1)), dim=1
        ),
        dim=1,
    )
    loss_sm = 1e-5 * loss_smooth(disp12, im1)
    loss = loss_warp + loss_sm

    return loss.mean()


def triplet_loss(
    disp12, im1, im2, mask, wsize, fakedisp12=None, margin=0.1, metric="l2"
):
    md = (wsize - 1) // 2
    p = disp_warp(im2, disp12, False)  # Need to sample from right side of img1
    if fakedisp12 is not None:
        n = disp_warp(im1, fakedisp12, False)
    else:
        choices = np.array([[x, -x] for x in range(wsize, im1.size(-1) // 2)]).flatten()
        n = torch.roll(im1, np.random.choice(choices), dims=-1)

    sp = nn.Softplus()

    if metric == "l1":
        triloss = sp((im1 - p).abs() - (im1 - n).abs() + margin)
    elif metric == "l2":
        triloss = sp((im1 - p) ** 2 - (im1 - n) ** 2 + margin)
    elif metric == "ssim":
        triloss = sp(SSIM(im1, p, md) - SSIM(im1, n, md) + margin)
    elif metric == "csim":
        triloss = sp(CSIM(im1, p, md) - CSIM(im1, n, md) + margin)
    else:
        raise Exception(f"{metric} not implemented yet.")

    if metric in ["l1", "l2"]:
        refl = nn.ReflectionPad2d(md)
        triloss = nn.AvgPool2d(wsize, 1, 0)(refl(triloss))

    return triloss[mask > 0].mean()


def sparsity_loss(im1, im2, validhints, wvalidhints, wsize, metric="csim"):
    md = (wsize - 1) // 2
    blur_validhints = gaussian_blur(validhints * 255, wsize) / 255
    blur_wvalidhints = gaussian_blur(wvalidhints * 255, wsize) / 255

    if metric == "l1":
        loss = (im1 - blur_validhints).abs() + (im2 - blur_wvalidhints).abs()
    elif metric == "l2":
        loss = (im1 - blur_validhints) ** 2 + (im2 - blur_wvalidhints) ** 2
    elif metric == "ssim":
        loss = SSIM(im1, blur_validhints, md) + SSIM(im2, blur_wvalidhints, md)
    elif metric == "csim":
        loss = CSIM(im1, blur_validhints, md) + CSIM(im2, blur_wvalidhints, md)
    else:
        raise Exception(f"{metric} not implemented yet.")

    assert im1.shape[-2:] == loss.shape[-2:]

    return loss, blur_validhints, blur_wvalidhints


def total_variation_loss(img, weight=1.0):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[..., 1:, :] - img[..., :-1, :], 2).sum()
    tv_w = torch.pow(img[..., :, 1:] - img[..., :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


def middlebury_metrics(disp, gt, valid):
    error = np.abs(disp - gt)
    error[valid == 0] = 0
    q = np.array([50, 75, 99])

    bad05 = (error[valid > 0] > 0.5).astype(np.float32).mean()
    bad1 = (error[valid > 0] > 1.0).astype(np.float32).mean()
    bad2 = (error[valid > 0] > 2.0).astype(np.float32).mean()
    bad4 = (error[valid > 0] > 4.0).astype(np.float32).mean()
    avgerr = error[valid > 0].mean()
    rms = (disp - gt) ** 2
    rms = np.sqrt(rms[valid > 0].mean())
    a50, a90, a95, a99 = (
        np.percentile(error, 50),
        np.percentile(error, 90),
        np.percentile(error, 95),
        np.percentile(error, 99),
    )
    return {
        "bad 0.5": bad05,
        "bad 1.0": bad1,
        "bad 2.0": bad2,
        "bad 4.0": bad4,
        "avgerr": avgerr,
        "rms": rms,
        "A50": a50,
        "A90": a90,
        "A95": a95,
        "A99": a99,
        "errormap": error * (valid > 0),
    }


def booster_metrics(disp, gt, valid):
    error = np.abs(disp - gt)
    error[valid == 0] = 0
    bad2 = (error[valid > 0] > 2.0).astype(np.float32).mean()
    bad4 = (error[valid > 0] > 4.0).astype(np.float32).mean()
    bad6 = (error[valid > 0] > 6.0).astype(np.float32).mean()
    bad8 = (error[valid > 0] > 8.0).astype(np.float32).mean()

    avgerr = error[valid > 0].mean()
    rms = (disp - gt) ** 2
    rms = np.sqrt(rms[valid > 0].mean())
    return {
        "bad 2.0": bad2,
        "bad 4.0": bad4,
        "bad 6.0": bad6,
        "bad 8.0": bad8,
        "avgerr": avgerr,
        "rms": rms,
        "errormap": error * (valid > 0),
    }


def kitti_metrics(disp, gt, valid):
    error = np.abs(disp - gt)

    bad3 = (
        ((error[valid > 0] > 3) * (error[valid > 0] / gt[valid > 0] > 0.05))
        .astype(np.float32)
        .mean()
    )
    avgerr = error[valid > 0].mean()
    return {"bad 3": bad3, "epe": avgerr, "errormap": error * (valid > 0)}


def sample_hints(hints, validhints, probability=0.20):
    new_validhints = (
        validhints * (torch.rand_like(validhints, dtype=torch.float32) < probability)
    ).float()
    new_hints = hints * new_validhints  # zero invalid hints
    new_hints[new_validhints == 0] = 0
    # new_hints[new_hints>5000] = 0
    return new_hints, new_validhints


def depth_metrics(depth, gt_depth, valid):
    error = np.abs(depth - gt_depth)
    rms = (depth - gt_depth) ** 2

    error[valid == 0] = 0
    rms[valid == 0] = 0

    thresh = np.maximum((gt_depth / depth), (depth / gt_depth))

    a1_105 = (thresh[valid > 0] < 1.05).astype(np.float32).mean()
    a2_105 = (thresh[valid > 0] < 1.05**2).astype(np.float32).mean()
    a3_105 = (thresh[valid > 0] < 1.05**3).astype(np.float32).mean()

    a1_115 = (thresh[valid > 0] < 1.15).astype(np.float32).mean()
    a2_115 = (thresh[valid > 0] < 1.15**2).astype(np.float32).mean()
    a3_115 = (thresh[valid > 0] < 1.15**3).astype(np.float32).mean()

    a1_125 = (thresh[valid > 0] < 1.25).astype(np.float32).mean()
    a2_125 = (thresh[valid > 0] < 1.25**2).astype(np.float32).mean()
    a3_125 = (thresh[valid > 0] < 1.25**3).astype(np.float32).mean()

    avgerr = error[valid > 0].mean()
    rmserr = np.sqrt(rms[valid > 0].mean())

    avgrelerr = (error[valid > 0] / gt_depth[valid > 0]).mean()

    all = {
        "a1_105": a1_105 * 100,
        "a2_105": a2_105 * 100,
        "a3_105": a3_105 * 100,
        "a1_115": a1_115 * 100,
        "a2_115": a2_115 * 100,
        "a3_115": a3_115 * 100,
        "a1_125": a1_125 * 100,
        "a2_125": a2_125 * 100,
        "a3_125": a3_125 * 100,
        "avgerr": avgerr,
        "rms": rmserr,
        "avgrelerr": avgrelerr * 100,
        "errormap": error * (valid > 0),
    }

    return all


def guided_metrics(disp, gt, valid, maskocc=None):
    error = np.abs(disp - gt)
    rms = (disp - gt) ** 2

    error[valid == 0] = 0
    rms[valid == 0] = 0

    bad1 = (error[valid > 0] > 1.0).astype(np.float32).mean()
    bad2 = (error[valid > 0] > 2.0).astype(np.float32).mean()
    bad3 = (error[valid > 0] > 3.0).astype(np.float32).mean()
    bad4 = (error[valid > 0] > 4.0).astype(np.float32).mean()
    bad5 = (error[valid > 0] > 5.0).astype(np.float32).mean()
    bad6 = (error[valid > 0] > 6.0).astype(np.float32).mean()
    bad7 = (error[valid > 0] > 7.0).astype(np.float32).mean()
    bad8 = (error[valid > 0] > 8.0).astype(np.float32).mean()
    avgerr = error[valid > 0].mean()
    rmserr = np.sqrt(rms[valid > 0].mean())

    all = {
        "bad 1.0": bad1,
        "bad 2.0": bad2,
        "bad 3.0": bad3,
        "bad 4.0": bad4,
        "bad 5.0": bad5,
        "bad 6.0": bad6,
        "bad 7.0": bad7,
        "bad 8.0": bad8,
        "avgerr": avgerr,
        "rms": rmserr,
        "errormap": error * (valid > 0),
    }

    if maskocc is not None and maskocc.sum() != 0:
        error_occ = np.copy(error)
        error_occ = error_occ[(maskocc > 0) & (valid > 0)]
        rms_occ = np.copy(rms)
        rms_occ = rms_occ[(maskocc > 0) & (valid > 0)]

        bad1 = (error_occ > 1.0).astype(np.float32).mean()
        bad2 = (error_occ > 2.0).astype(np.float32).mean()
        bad3 = (error_occ > 3.0).astype(np.float32).mean()
        bad4 = (error_occ > 4.0).astype(np.float32).mean()
        bad5 = (error_occ > 5.0).astype(np.float32).mean()
        bad6 = (error_occ > 6.0).astype(np.float32).mean()
        bad7 = (error_occ > 7.0).astype(np.float32).mean()
        bad8 = (error_occ > 8.0).astype(np.float32).mean()
        avgerr = error_occ.mean()
        rmserr = np.sqrt(rms_occ.mean())

        occ = {
            "occ bad 1.0": bad1,
            "occ bad 2.0": bad2,
            "occ bad 3.0": bad3,
            "occ bad 4.0": bad4,
            "occ bad 5.0": bad5,
            "occ bad 6.0": bad6,
            "occ bad 7.0": bad7,
            "occ bad 8.0": bad8,
            "occ avgerr": avgerr,
            "occ rms": rmserr,
        }
    else:
        occ = {
            "occ bad 1.0": np.nan,
            "occ bad 2.0": np.nan,
            "occ bad 3.0": np.nan,
            "occ bad 4.0": np.nan,
            "occ bad 5.0": np.nan,
            "occ bad 6.0": np.nan,
            "occ bad 7.0": np.nan,
            "occ bad 8.0": np.nan,
            "occ avgerr": np.nan,
            "occ rms": 0,
        }
        # occ = {'occ bad 1.0':-1, 'occ bad 2.0':-1, 'occ bad 3.0':-1, 'occ bad 4.0':-1, 'occ bad 5.0':-1, 'occ bad 6.0':-1, 'occ bad 7.0':-1, 'occ bad 8.0':-1, 'occ avgerr':-1, 'occ rms':-1}

    if maskocc is not None and maskocc.sum() != 0:

        error_noc = error.copy()
        error_noc = error_noc[(maskocc == 0) & (valid > 0)]
        rms_noc = rms.copy()
        rms_noc = rms_noc[(maskocc == 0 & (valid > 0))]

        bad1 = (error_noc > 1.0).astype(np.float32).mean()
        bad2 = (error_noc > 2.0).astype(np.float32).mean()
        bad3 = (error_noc > 3.0).astype(np.float32).mean()
        bad4 = (error_noc > 4.0).astype(np.float32).mean()
        bad5 = (error_noc > 5.0).astype(np.float32).mean()
        bad6 = (error_noc > 6.0).astype(np.float32).mean()
        bad7 = (error_noc > 7.0).astype(np.float32).mean()
        bad8 = (error_noc > 8.0).astype(np.float32).mean()
        avgerr = error_noc.mean()
        rmserr = np.sqrt(rms_noc.mean())

        noc = {
            "noc bad 1.0": bad1,
            "noc bad 2.0": bad2,
            "noc bad 3.0": bad3,
            "noc bad 4.0": bad4,
            "noc bad 5.0": bad5,
            "noc bad 6.0": bad6,
            "noc bad 7.0": bad7,
            "noc bad 8.0": bad8,
            "noc avgerr": avgerr,
            "noc rms": rmserr,
        }
    else:
        noc = {
            "noc bad 1.0": bad1,
            "noc bad 2.0": bad2,
            "noc bad 3.0": bad3,
            "noc bad 4.0": bad4,
            "noc bad 5.0": bad5,
            "noc bad 6.0": bad6,
            "noc bad 7.0": bad7,
            "noc bad 8.0": bad8,
            "noc avgerr": avgerr,
            "noc rms": rmserr,
        }
        # noc = {'noc bad 1.0':-1, 'noc bad 2.0':-1, 'noc bad 3.0':-1, 'noc bad 4.0':-1, 'noc bad 5.0':-1, 'noc bad 6.0':-1, 'noc bad 7.0':-1, 'noc bad 8.0':-1, 'noc avgerr':-1, 'noc rms':-1}

    concat = dict(all)
    concat.update(occ)
    concat.update(noc)

    return concat


# From DepthAnythingV2
class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
        )

        return loss


class AffineInvariantMAELoss(nn.Module):
    def __init__(self):
        super(AffineInvariantMAELoss, self).__init__()

    def forward(
        self,
        predicted_depth,
        target_depth,
        valid_mask,
        min_quantile=0.02,
        max_quantile=0.98,
    ):
        """
        Compute the affine-invariant MAE loss between predicted and target depth maps.

        :param predicted_depth: Predicted depth map.
        :param target_depth: Ground truth depth map.
        :return: Affine-invariant MAE loss.
        """

        # Compute the minimum and maximum values for scaling
        min_pred = torch.quantile(predicted_depth[valid_mask], min_quantile)
        max_pred = torch.quantile(predicted_depth[valid_mask], max_quantile)
        min_target = torch.quantile(target_depth[valid_mask], min_quantile)
        max_target = torch.quantile(target_depth[valid_mask], max_quantile)

        # Scale predicted depth to match target depth range
        scaled_pred = ((predicted_depth - min_pred) / (max_pred - min_pred)) * (
            max_target - min_target
        ) + min_target

        # Compute affine-invariant MAE
        loss = torch.mean(torch.abs(scaled_pred - target_depth)[valid_mask])

        return loss


class AffineInvariantV2MAELoss(nn.Module):
    def __init__(self):
        super(AffineInvariantV2MAELoss, self).__init__()

    def forward(
        self,
        predicted_depth,
        target_depth,
        valid_mask,
        min_quantile=0.02,
        max_quantile=0.98,
    ):
        """
        Compute the affine-invariant MAE loss between predicted and target depth maps.

        :param predicted_depth: Predicted depth map.
        :param target_depth: Ground truth depth map.
        :return: Affine-invariant MAE loss.
        """

        # Compute the minimum and maximum values for scaling
        _target_depth_t = torch.median(target_depth[valid_mask])
        _target_depth_s = torch.mean(
            torch.abs(target_depth[valid_mask] - _target_depth_t)
        )
        _target_depth = (target_depth - _target_depth_t) / _target_depth_s

        _predicted_depth_t = torch.median(predicted_depth[valid_mask])
        _predicted_depth_s = torch.mean(
            torch.abs(predicted_depth[valid_mask] - _predicted_depth_t)
        )
        _predicted_depth = (predicted_depth - _predicted_depth_t) / _predicted_depth_s

        # Compute affine-invariant MAE
        loss = torch.mean(torch.abs(_predicted_depth - _target_depth)[valid_mask])

        return loss
