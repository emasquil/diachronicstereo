import random
import torch
import torch.nn.functional as F
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


class StereoAugmentor:
    def __init__(
        self,
        crop_size,
        min_scale=-0.2,
        max_scale=0.5,
        stretch_prob=0.8,
        max_stretch=0.2,
        do_flip=True,
        h_flip_prob=0.5,
        v_flip_prob=0.1,
        yjitter=False,
        saturation_range=(0.6, 1.4),
        brightness=0.4,
        contrast=0.4,
        hue=0.159,
        gamma_range=(1.0, 1.0, 1.0, 1.0),
        asymmetric_prob=0.2,
        eraser_prob=0.5,
    ):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.stretch_prob = stretch_prob
        self.max_stretch = max_stretch
        self.do_flip = do_flip
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.yjitter = yjitter
        self.asymmetric_prob = asymmetric_prob
        self.gamma_range = gamma_range
        self.eraser_prob = eraser_prob

        self.color_jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=tuple(saturation_range),
            hue=hue,
        )

    def random_scaling_and_stretch(self, imgs, flows=None):
        scale = 2 ** random.uniform(self.min_scale, self.max_scale)
        scale_x, scale_y = scale, scale
        if random.random() < self.stretch_prob:
            scale_x *= 2 ** random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** random.uniform(-self.max_stretch, self.max_stretch)

        nh, nw = int(imgs[0].shape[-2] * scale_y), int(imgs[0].shape[-1] * scale_x)
        imgs = [
            TF.resize(img, (nh, nw), interpolation=TF.InterpolationMode.BILINEAR)
            for img in imgs
        ]

        if flows is not None:
            flows = [
                F.interpolate(
                    flow.unsqueeze(0),
                    size=(nh, nw),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                * scale_x
                for flow in flows
            ]

        return imgs, flows

    def random_flip(self, imgs, flows=None):
        if not self.do_flip:
            return imgs, flows
        if random.random() < self.h_flip_prob:
            imgs = [TF.hflip(img) for img in imgs]
            if flows is not None:
                flows = [TF.hflip(flow) * -1 for flow in flows]

        if random.random() < self.v_flip_prob:
            imgs = [TF.vflip(img) for img in imgs]
            if flows is not None:
                flows = [TF.vflip(flow) for flow in flows]

        return imgs, flows

    def random_crop(self, imgs, flows=None):
        h, w = imgs[0].shape[-2:]
        th, tw = self.crop_size, self.crop_size

        if h < th or w < tw:
            pad_h = max(th - h, 0)
            pad_w = max(tw - w, 0)
            imgs = [F.pad(img, (0, pad_w, 0, pad_h), mode="reflect") for img in imgs]
            if flows:
                flows = [
                    F.pad(flow, (0, pad_w, 0, pad_h), mode="reflect") for flow in flows
                ]
            h, w = imgs[0].shape[-2:]

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        if self.yjitter and random.random() < 0.5:
            y_shift = random.randint(-2, 2)
            imgs[0] = imgs[0][:, i : i + th, j : j + tw]
            imgs[1] = imgs[1][:, i + y_shift : i + th + y_shift, j : j + tw]
        else:
            imgs = [img[:, i : i + th, j : j + tw] for img in imgs]

        if flows:
            flows = [flow[:, i : i + th, j : j + tw] for flow in flows]

        return imgs, flows

    def photometric_aug(self, img1, img2):
        if random.random() < self.asymmetric_prob:
            img1 = self.color_jitter(img1)
            img2 = self.color_jitter(img2)
        else:
            stack = torch.stack([img1, img2])
            stack = self.color_jitter(stack)
            img1, img2 = stack[0], stack[1]

        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        gain = random.uniform(self.gamma_range[2], self.gamma_range[3])
        img1 = TF.adjust_gamma(img1, gamma, gain).clamp(0, 1)
        img2 = TF.adjust_gamma(img2, gamma, gain).clamp(0, 1)
        return img1, img2

    def eraser_aug(self, img):
        if random.random() < self.eraser_prob:
            _, h, w = img.shape
            mean_color = img.mean(dim=[1, 2], keepdim=True)
            for _ in range(random.randint(1, 3)):
                x0 = random.randint(0, w)
                y0 = random.randint(0, h)
                dx = random.randint(50, 100)
                dy = random.randint(50, 100)
                x1 = min(w, x0 + dx)
                y1 = min(h, y0 + dy)
                img[:, y0:y1, x0:x1] = mean_color
        return img

    def __call__(self, sample):
        imgs = [sample["left"], sample["right"]]

        # collect all disparity-like maps present
        flow_keys, flows = [], []
        for k in ("disparity", "disparity_no_trees"):
            if k in sample:
                flow_keys.append(k)
                flows.append(sample[k])
        flows = flows if flows else None

        # spatial ops (identical randomness for imgs + all flows)
        imgs, flows = self.random_scaling_and_stretch(imgs, flows)
        imgs, flows = self.random_flip(imgs, flows)
        imgs, flows = self.random_crop(imgs, flows)

        # photometric (images only)
        img1, img2 = imgs
        img1, img2 = self.photometric_aug(img1, img2)
        img2 = self.eraser_aug(img2)

        # write back
        sample["left"], sample["right"] = img1, img2
        if flows is not None:
            for k, f in zip(flow_keys, flows):
                sample[k] = f
        return sample
