from .base_dataset import BaseDataset
from glob import glob
import os.path as osp
from .frame_utils import *
import numpy as np
import os
import cv2
import torch


class KITTIStereoDataset(BaseDataset):
    def load_data(self, datapath):
        image_list = sorted(glob(osp.join(datapath, "image_2/*_10.png")))

        # Please make symbolic links to reproduce the correct paths

        for i in range(len(image_list)):
            _tmp = [
                image_list[i].replace("image_2", "disp_occ_0"),
                image_list[i],
                image_list[i].replace("image_2", "image_3"),
                image_list[i].replace("image_2", "disp_noc_0"),
            ]

            if self.mono is not None:
                _tmp += [
                    image_list[i].replace("image_2", f"{self.mono}_2"),
                    image_list[i].replace("image_2", f"{self.mono}_3"),
                ]

            self.image_list += [_tmp]
            self.extra_info += [image_list[i].split("/")[-1]]  # scene and frame_id

    def load_sample(self, index):
        data = {}

        data["im2"] = np.array(read_gen(self.image_list[index][1])).astype(np.uint8)
        data["im3"] = np.array(read_gen(self.image_list[index][2])).astype(np.uint8)

        if self.mono is not None:
            data["im2_mono"] = np.expand_dims(
                read_mono(self.image_list[index][4]).astype(np.float32), -1
            )
            data["im3_mono"] = np.expand_dims(
                read_mono(self.image_list[index][5]).astype(np.float32), -1
            )

        if self.is_test:
            data["im2"] = data["im2"] / 255.0
            data["im3"] = data["im3"] / 255.0

        # grayscale images
        data["im2"] = self.gray2rgb(data["im2"])
        data["im3"] = self.gray2rgb(data["im3"])

        if os.path.exists(self.image_list[index][0]):
            data["gt"], data["validgt"] = readDispKITTI(self.image_list[index][0])
        else:
            data["gt"] = np.zeros_like(data["im2"])[:, :, 0:1]
            data["validgt"] = np.zeros_like(data["im2"], dtype=np.uint8)[:, :, 0:1]

        if os.path.exists(self.image_list[index][3]):
            _, data["maskocc"] = readDispKITTI(self.image_list[index][3])
        else:
            data["maskocc"] = np.zeros_like(data["validgt"])

        if self.is_test:
            data["maskocc"] = (data["validgt"] > 0) & (
                data["validgt"] - data["maskocc"] > 0
            )  # 1 if occluded, 0 otherwise
            data["maskocc"] = np.array(data["maskocc"]).astype(np.uint8)
        else:
            data["maskocc"] = None

        data["gt"] = np.array(data["gt"]).astype(np.float32)
        data["validgt"] = np.array(data["validgt"]).astype(np.uint8)

        data["gt_right"] = np.zeros_like(data["gt"])
        data["validgt_right"] = (data["gt_right"] > 0).astype(np.uint8)

        data["gt_right"] = np.array(data["gt_right"]).astype(np.float32)
        data["validgt_right"] = np.array(data["validgt_right"]).astype(np.uint8)

        if self.top_crop > 0:
            for key in data:
                data[key] = data[key][self.top_crop :, ...]

        # TODO: check resize
        # for k in data:
        #     if data[k] is not None:
        #         if k not in ['gt', 'gt_right', 'validgt', 'validgt_right', 'maskocc']:
        #             data[k] = cv2.resize(data[k], (int(data[k].shape[1]/self.scale_factor), int(data[k].shape[0]/self.scale_factor)), interpolation=cv2.INTER_LINEAR)
        #         else:
        #             data[k] = cv2.resize(data[k], (int(data[k].shape[1]/self.scale_factor), int(data[k].shape[0]/self.scale_factor)), interpolation=cv2.INTER_NEAREST)

        #         if len(data[k].shape) == 2:
        #             data[k] = np.expand_dims(data[k], -1)

        #         if k in ['gt', 'gt_right']:
        #             data[k] = data[k] / self.scale_factor

        data = self.rescale_data(data)

        if self.is_test or self.augmentor is None:
            augm_data = data
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

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float()

        augm_data["extra_info"] = self.extra_info[index]

        return augm_data
