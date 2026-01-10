from .base_dataset import BaseDataset
from glob import glob
import os.path as osp
from .frame_utils import *
import numpy as np
import os
import cv2
import torch
import pickle


class LayeredFlowDataset(BaseDataset):
    def load_data(self, datapath):
        scene_list = sorted(
            glob(osp.join(datapath) + "/*"), key=lambda x: int(x.split("/")[-1])
        )

        for i in range(len(scene_list)):
            gt = "%s/gt.pickle" % (scene_list[i])

            left_img_0 = "%s/0_0.png" % (scene_list[i])
            right_img_0 = "%s/0_1.png" % (scene_list[i])

            self.image_list += [[left_img_0, right_img_0, gt, [0, 1]]]
            self.extra_info += [f"{i}_0"]  # scene and frame_id

            left_img_1 = "%s/3_0.png" % (scene_list[i])
            right_img_1 = "%s/3_1.png" % (scene_list[i])

            self.image_list += [[left_img_1, right_img_1, gt, [2, 3]]]
            self.extra_info += [f"{i}_3"]  # scene and frame_id

    def load_sample(self, index):
        data = {}

        if not self.is_test:
            raise NotImplementedError

        data["im2"] = (
            np.array(read_gen(self.image_list[index][0])).astype(np.uint8) / 255.0
        )
        data["im3"] = (
            np.array(read_gen(self.image_list[index][1])).astype(np.uint8) / 255.0
        )

        # grayscale images
        data["im2"] = self.gray2rgb(data["im2"])
        data["im3"] = self.gray2rgb(data["im3"])

        data["gt"] = np.zeros_like(data["im2"], dtype=np.float32)[..., 0]

        gt = pickle.load(open(self.image_list[index][2], "rb"))
        stereo_points = gt["stereo_points"]
        stereo_annotations = gt["annotations"]
        stereo_ids = self.image_list[index][3]

        for point1, point2 in zip(
            stereo_points[stereo_ids[0]], stereo_points[stereo_ids[1]]
        ):
            point1_index = (point1[0], point1[1])
            point2_index = (point2[0], point2[1])

            x1, y1 = point1[2]
            x2, y2 = point2[2]

            if point1_index == point2_index:
                # Use only the first layer of the GT
                if stereo_annotations[point1_index][2] == 0:
                    # Stereo pairs with significant y-axis discrepancies are excluded
                    if abs(y2 - y1) <= 2:
                        x1, y1 = int(x1), int(y1)
                        data["gt"][y1, x1] = np.linalg.norm([x2 - x1, y2 - y1])

        data["validgt"] = data["gt"] > 0

        data["gt"] = np.expand_dims(data["gt"], -1)
        data["validgt"] = np.expand_dims(data["validgt"], -1)

        data["im2"] = torch.from_numpy(data["im2"]).permute(2, 0, 1).float()
        data["im3"] = torch.from_numpy(data["im3"]).permute(2, 0, 1).float()
        data["gt"] = torch.from_numpy(data["gt"]).permute(2, 0, 1).float()
        data["validgt"] = torch.from_numpy(data["validgt"]).permute(2, 0, 1).float()

        data["extra_info"] = self.extra_info[index]

        return data
