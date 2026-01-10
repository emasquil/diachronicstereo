import numpy as np
import random
import math
from PIL import Image
import albumentations as A

from numba import njit

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F


class DisparityAugmentor:
    def __init__(
        self, crop_size, min_scale=-0.2, max_scale=0.5, asym=0.3, do_flip=True
    ):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.crop_prob = 1

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.1
        self.v_flip_prob = 0.1
        self.all_image_prob = 0.0

        # photometric augmentation params
        # self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)

        self.photo_aug = A.Compose(
            [
                # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14, p=0.5),
                A.RGBShift(p=0.5),
                A.ChannelDropout(p=0.1),
                A.Equalize(p=0.1),
                A.HueSaturationValue(p=0.5),
                A.ChannelShuffle(p=0.2),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=0.1),
                A.MotionBlur(p=0.1),
                A.Blur(p=0.2),
                A.ToGray(p=0.1),
                A.MedianBlur(p=0.1),
                A.ImageCompression(p=0.1),
                A.GaussNoise(p=0.1),
                A.GaussianBlur(p=0.1),
                A.CLAHE(p=0.1),
            ],
            p=1,
        )

        self.asymmetric_color_aug_prob = asym

    def color_transform(self, im2, im3):
        """Photometric augmentation"""

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            # im2 = np.array(self.photo_aug(Image.fromarray(im2)), dtype=np.uint8)
            # im3 = np.array(self.photo_aug(Image.fromarray(im3)), dtype=np.uint8)
            im2 = self.photo_aug(image=im2)["image"]
            im3 = self.photo_aug(image=im3)["image"]

        # symmetric
        else:
            image_stack = np.concatenate([im2, im3], axis=0)
            image_stack = self.photo_aug(image=image_stack)["image"]
            im2, im3 = np.split(image_stack, 2, axis=0)

        return im2, im3

    def spatial_transform(
        self,
        im2,
        im3,
        im2_mono=None,
        im3_mono=None,
        gt2=None,
        validgt2=None,
        gt3=None,
        validgt3=None,
        maskocc=None,
    ):
        # randomly sample scale
        ht, wd = im2.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd)
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            im2 = cv2.resize(
                im2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            im3 = cv2.resize(
                im3, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )

            if gt2 is not None:
                gt2 = np.expand_dims(
                    cv2.resize(
                        gt2,
                        None,
                        fx=scale_x,
                        fy=scale_y,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    * scale_x,
                    -1,
                )
                validgt2 = np.expand_dims(
                    cv2.resize(
                        validgt2,
                        None,
                        fx=scale_x,
                        fy=scale_y,
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    -1,
                )

            if gt3 is not None:
                gt3 = np.expand_dims(
                    cv2.resize(
                        gt3,
                        None,
                        fx=scale_x,
                        fy=scale_y,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    * scale_x,
                    -1,
                )
                validgt3 = np.expand_dims(
                    cv2.resize(
                        validgt3,
                        None,
                        fx=scale_x,
                        fy=scale_y,
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    -1,
                )

            if im2_mono is not None:
                im2_mono = np.expand_dims(
                    cv2.resize(
                        im2_mono,
                        None,
                        fx=scale_x,
                        fy=scale_y,
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    -1,
                )

            if im3_mono is not None:
                im3_mono = np.expand_dims(
                    cv2.resize(
                        im3_mono,
                        None,
                        fx=scale_x,
                        fy=scale_y,
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    -1,
                )

            if maskocc is not None:
                maskocc = np.expand_dims(
                    cv2.resize(
                        maskocc,
                        None,
                        fx=scale_x,
                        fy=scale_y,
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    -1,
                )

        # disabled, to be implemented
        if self.do_flip:
            if (
                np.random.rand() < self.h_flip_prob
                and gt2 is not None
                and gt3 is not None
            ):  # h-flip
                tmp = im2[:, ::-1]
                im2 = im3[:, ::-1]
                im3 = tmp

                tmp = gt2[:, ::-1]
                gt2 = gt3[:, ::-1]
                gt3 = tmp

                tmp = validgt2[:, ::-1]
                validgt2 = validgt3[:, ::-1]
                validgt3 = tmp

                if im2_mono is not None and im3_mono is not None:
                    tmp = im2_mono[:, ::-1]
                    im2_mono = im3_mono[:, ::-1]
                    im3_mono = tmp

                # not correct, however, not used during training
                if maskocc is not None:
                    maskocc = maskocc[:, ::-1]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                im2 = np.flip(im2, axis=0)
                im3 = np.flip(im3, axis=0)
                if gt2 is not None:
                    gt2 = np.flip(gt2, axis=0)
                    validgt2 = np.flip(validgt2, axis=0)

                if gt3 is not None:
                    gt3 = np.flip(gt3, axis=0)
                    validgt3 = np.flip(validgt3, axis=0)

                if im2_mono is not None:
                    im2_mono = np.flip(im2_mono, axis=0)

                if im3_mono is not None:
                    im3_mono = np.flip(im3_mono, axis=0)

                if maskocc is not None:
                    maskocc = np.flip(maskocc, axis=0)

        # resize images to crop size
        if np.random.rand() < self.all_image_prob:
            im2 = cv2.resize(
                im2,
                (self.crop_size[1], self.crop_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            im3 = cv2.resize(
                im3,
                (self.crop_size[1], self.crop_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            if gt2 is not None:
                scale_disp = gt2.shape[1] / self.crop_size[1]
                gt2 = np.expand_dims(
                    cv2.resize(
                        gt2,
                        (self.crop_size[1], self.crop_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    / scale_disp,
                    -1,
                )
                validgt2 = np.expand_dims(
                    cv2.resize(
                        validgt2,
                        (self.crop_size[1], self.crop_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    -1,
                )

            if gt3 is not None:
                scale_disp = gt3.shape[1] / self.crop_size[1]
                gt3 = np.expand_dims(
                    cv2.resize(
                        gt3,
                        (self.crop_size[1], self.crop_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    / scale_disp,
                    -1,
                )
                validgt3 = np.expand_dims(
                    cv2.resize(
                        validgt3,
                        (self.crop_size[1], self.crop_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    -1,
                )

            if im2_mono is not None:
                im2_mono = np.expand_dims(
                    cv2.resize(
                        im2_mono,
                        (self.crop_size[1], self.crop_size[0]),
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    -1,
                )

            if im3_mono is not None:
                im3_mono = np.expand_dims(
                    cv2.resize(
                        im3_mono,
                        (self.crop_size[1], self.crop_size[0]),
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    -1,
                )

            if maskocc is not None:
                maskocc = np.expand_dims(
                    cv2.resize(
                        maskocc,
                        (self.crop_size[1], self.crop_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    -1,
                )

        # allow full size crops
        if (im2.shape[0] - self.crop_size[0] > 0) and np.random.rand() < self.crop_prob:
            y0 = np.random.randint(0, im2.shape[0] - self.crop_size[0])
        else:
            y0 = 0

        if (im2.shape[1] - self.crop_size[1] > 0) and np.random.rand() < self.crop_prob:
            x0 = np.random.randint(0, im2.shape[1] - self.crop_size[1])
        else:
            x0 = 0

        im2 = im2[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        im3 = im3[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        if gt2 is not None:
            gt2 = gt2[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            validgt2 = validgt2[
                y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
            ]

        if gt3 is not None:
            gt3 = gt3[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            validgt3 = validgt3[
                y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
            ]

        if im2_mono is not None:
            im2_mono = im2_mono[
                y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
            ]

        if im3_mono is not None:
            im3_mono = im3_mono[
                y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
            ]

        if maskocc is not None:
            maskocc = maskocc[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        # Ensure that all the images respect crop size -- i.e., image size >= crop size
        if self.crop_size[0] - im2.shape[0] > 0 or self.crop_size[1] - im2.shape[1] > 0:
            _newW = max(self.crop_size[1], im2.shape[1])
            _newH = max(self.crop_size[0], im2.shape[0])
            _scale_factor = _newW / im2.shape[1]

            im2 = cv2.resize(im2, (_newW, _newH), interpolation=cv2.INTER_LINEAR)
            im3 = cv2.resize(im3, (_newW, _newH), interpolation=cv2.INTER_LINEAR)

            if gt2 is not None:
                gt2 = np.expand_dims(
                    cv2.resize(gt2, (_newW, _newH), interpolation=cv2.INTER_NEAREST)
                    * _scale_factor,
                    -1,
                )
                validgt2 = np.expand_dims(
                    cv2.resize(
                        validgt2, (_newW, _newH), interpolation=cv2.INTER_NEAREST
                    ),
                    -1,
                )

            if gt3 is not None:
                gt3 = np.expand_dims(
                    cv2.resize(gt3, (_newW, _newH), interpolation=cv2.INTER_NEAREST)
                    * _scale_factor,
                    -1,
                )
                validgt3 = np.expand_dims(
                    cv2.resize(
                        validgt3, (_newW, _newH), interpolation=cv2.INTER_NEAREST
                    ),
                    -1,
                )

            if im2_mono is not None:
                im2_mono = np.expand_dims(
                    cv2.resize(
                        im2_mono, (_newW, _newH), interpolation=cv2.INTER_LINEAR
                    ),
                    -1,
                )

            if im3_mono is not None:
                im3_mono = np.expand_dims(
                    cv2.resize(
                        im3_mono, (_newW, _newH), interpolation=cv2.INTER_LINEAR
                    ),
                    -1,
                )

            if maskocc is not None:
                maskocc = np.expand_dims(
                    cv2.resize(
                        maskocc, (_newW, _newH), interpolation=cv2.INTER_NEAREST
                    ),
                    -1,
                )

        return im2, im3, im2_mono, im3_mono, gt2, validgt2, gt3, validgt3, maskocc

    def __call__(
        self,
        im2,
        im3,
        im2_mono=None,
        im3_mono=None,
        gt2=None,
        validgt2=None,
        gt3=None,
        validgt3=None,
        maskocc=None,
    ):
        im2c, im3c = self.color_transform(im2, im3)
        im2, im3, im2_mono, im3_mono, gt2, validgt2, gt3, validgt3, maskocc = (
            self.spatial_transform(
                np.concatenate((im2, im2c), -1),
                np.concatenate((im3, im3c), -1),
                im2_mono,
                im3_mono,
                gt2,
                validgt2,
                gt3,
                validgt3,
                maskocc,
            )
        )

        im2 = np.ascontiguousarray(im2) / 255.0  # - 1.
        im3 = np.ascontiguousarray(im3) / 255.0  # - 1.

        if im2_mono is not None:
            im2_mono = np.ascontiguousarray(im2_mono)

        if im3_mono is not None:
            im3_mono = np.ascontiguousarray(im3_mono)

        if gt2 is not None:
            gt2 = np.ascontiguousarray(gt2)
            validgt2 = np.ascontiguousarray(validgt2)

        if gt3 is not None:
            gt3 = np.ascontiguousarray(gt3)
            validgt3 = np.ascontiguousarray(validgt3)

        if maskocc is not None:
            maskocc = np.ascontiguousarray(maskocc)

        augm_data = {
            "im2": im2[:, :, :3],
            "im3": im3[:, :, :3],
            "im2_aug": im2[:, :, 3:6],
            "im3_aug": im3[:, :, 3:6],
        }

        if im2_mono is not None:
            augm_data["im2_mono"] = im2_mono

        if im3_mono is not None:
            augm_data["im3_mono"] = im3_mono

        if gt2 is not None:
            augm_data["gt"] = gt2
            augm_data["validgt"] = validgt2

        if gt3 is not None:
            augm_data["gt_right"] = gt3
            augm_data["validgt_right"] = validgt3

        if maskocc is not None:
            augm_data["maskocc"] = maskocc

        return augm_data
