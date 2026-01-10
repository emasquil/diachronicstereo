from .base_dataset import BaseDataset
from glob import glob
import os.path as osp
from .frame_utils import *
import numpy as np
import os
import cv2
import torch


class MonoTrapDataset(BaseDataset):
    def load_data(self, datapath):
        left_list = sorted(glob(os.path.join(datapath, "*/left_rect/*.png")))

        for i in range(len(left_list)):
            self.image_list += [
                [
                    left_list[i],
                    left_list[i].replace("left_rect", "right_rect"),
                    left_list[i].replace("left_rect", "gt_disp"),
                    left_list[i].replace("left_rect", "gt_depth"),
                    left_list[i].replace("left_rect", "gt_mask"),
                ]
            ]
            self.extra_info += [left_list[i].split("/")[-1]]  # scene and frame_id

    def load_sample(self, index):
        data = {}

        data["im2"] = np.array(read_gen(self.image_list[index][0])).astype(np.uint8)
        data["im3"] = np.array(read_gen(self.image_list[index][1])).astype(np.uint8)

        if self.is_test:
            data["im2"] = data["im2"] / 255.0
            data["im3"] = data["im3"] / 255.0

        # grayscale images
        data["im2"] = self.gray2rgb(data["im2"])
        data["im3"] = self.gray2rgb(data["im3"])

        _gt_mask = read_gen(self.image_list[index][4])
        _gt_mask = np.array(_gt_mask).astype(np.uint8)
        if len(_gt_mask.shape) == 3:
            _gt_mask = _gt_mask[..., 0]

        data["gt"], data["validgt"] = readDispKITTI(self.image_list[index][2])
        data["gt_depth"], data["validgt_depth"] = readDepthKITTI(
            self.image_list[index][3]
        )

        data["gt"] = np.array(data["gt"]).astype(np.float32)
        data["validgt"] = np.array(data["validgt"]).astype(np.uint8)

        data["gt_depth"] = np.array(data["gt_depth"]).astype(np.float32)
        data["validgt_depth"] = np.array(data["validgt_depth"]).astype(np.uint8)

        data["gt"][_gt_mask < 128] = 0
        data["validgt"][_gt_mask < 128] = 0

        data["gt_depth"][_gt_mask < 128] = 0
        data["validgt_depth"][_gt_mask < 128] = 0

        data["maskocc"] = np.expand_dims(np.zeros_like(data["validgt"]), -1)
        data["maskocc"] = np.array(data["maskocc"]).astype(np.uint8)

        if self.is_test:
            data["gt_right"] = np.zeros_like(data["gt"])
            data["validgt_right"] = (data["gt_right"] > 0).astype(np.uint8)
        else:
            raise ValueError("No right disparity available for training")

        data = self.rescale_data(data)

        if self.is_test or self.augmentor is None:
            data["im2_aug"] = data["im2"]
            data["im3_aug"] = data["im3"]
        else:
            im2_mono = data["im2_mono"] if self.mono is not None else None
            im3_mono = data["im3_mono"] if self.mono is not None else None
            augm_data = self.augmentor(
                data["im2"],
                data["im3"],
                im2_mono,
                im3_mono,
                gt2=data["gt"],
                validgt2=data["validgt"],
                maskocc=data["maskocc"],
                gt3=data["gt_right"],
                validgt3=data["validgt_right"],
            )

            for key in augm_data:
                data[key] = augm_data[key]

        for k in data:
            if data[k] is not None:
                data[k] = torch.from_numpy(data[k]).permute(2, 0, 1).float()

        data["extra_info"] = self.extra_info[index]

        return data
