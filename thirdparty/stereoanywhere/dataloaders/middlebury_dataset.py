from .base_dataset import BaseDataset
from glob import glob
import os.path as osp
from .frame_utils import *
import numpy as np
import os
import cv2
import torch


class MiddleburyDataset(BaseDataset):
    def load_data(self, datapath):
        image_list = sorted(glob(osp.join(datapath, "*/im0.png")))
        for i in range(len(image_list)):
            for im1 in ["im1", "im1E", "im1L"]:
                if os.path.exists(image_list[i].replace("im0", im1)):
                    _tmp = [
                        image_list[i].replace("im0.png", "disp0GT.pfm"),
                        image_list[i],
                        image_list[i].replace("im0", im1),
                        image_list[i].replace("im0.png", "mask0nocc.png"),
                        image_list[i].replace("im0.png", "disp1GT.pfm"),
                    ]

                    if self.mono is not None:
                        _tmp += [
                            image_list[i].replace("im0.png", f"im0_{self.mono}.png"),
                            image_list[i].replace("im0.png", f"{im1}_{self.mono}.png"),
                        ]

                    self.image_list += [_tmp]
                    self.extra_info += [
                        image_list[i].split("/")[-1]
                    ]  # scene and frame_id

    def load_sample(self, index):
        data = {}

        data["im2"] = np.array(read_gen(self.image_list[index][1])).astype(np.uint8)
        data["im3"] = np.array(read_gen(self.image_list[index][2])).astype(np.uint8)

        if self.mono is not None:
            data["im2_mono"] = np.expand_dims(
                read_mono(self.image_list[index][5]).astype(np.float32), -1
            )
            data["im3_mono"] = np.expand_dims(
                read_mono(self.image_list[index][6]).astype(np.float32), -1
            )

        if self.is_test:
            data["im2"] = data["im2"] / 255.0
            data["im3"] = data["im3"] / 255.0

        # grayscale images
        data["im2"] = self.gray2rgb(data["im2"])
        data["im3"] = self.gray2rgb(data["im3"])

        data["gt"] = np.expand_dims(readPFM(self.image_list[index][0]), -1)
        data["validgt"] = (data["gt"] < 5000) & (data["gt"] > 0)

        data["gt"] = np.array(data["gt"]).astype(np.float32)
        data["validgt"] = np.array(data["validgt"]).astype(np.uint8)

        if os.path.exists(self.image_list[index][4]):
            data["gt_right"] = np.expand_dims(readPFM(self.image_list[index][4]), -1)
            assert np.sum(data["gt_right"] > 0) > 0  # disabled for eth3d
        elif self.is_test:
            data["gt_right"] = np.zeros_like(data["gt"])
        else:
            raise ValueError(
                f"No right disparity available for training {self.image_list[index][4]}"
            )

        data["validgt_right"] = (data["gt_right"] < 5000) & (data["gt_right"] > 0)
        data["gt_right"] = np.array(data["gt_right"]).astype(np.float32)
        data["validgt_right"] = np.array(data["validgt_right"]).astype(np.uint8)

        if self.is_test:
            data["maskocc"] = np.expand_dims(
                np.array(read_gen(self.image_list[index][3])).astype(np.uint8), -1
            )
            data["maskocc"] = data["maskocc"] == 128  # 1 if occluded, 0 otherwise
            data["maskocc"] = np.array(data["maskocc"]).astype(np.uint8)
        else:
            data["maskocc"] = None

        data = self.rescale_data(data)

        if self.is_test or self.augmentor is None:
            data["im2_aug"] = data["im2"]
            data["im3_aug"] = data["im3"]
        else:
            im2_mono = data["im2_mono"] if self.mono is not None else None
            im3_mono = data["im3_mono"] if self.mono is not None else None
            data = self.augmentor(
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

        for k in data:
            if data[k] is not None:
                data[k] = torch.from_numpy(data[k]).permute(2, 0, 1).float()

        data["extra_info"] = self.extra_info[index]

        return data
