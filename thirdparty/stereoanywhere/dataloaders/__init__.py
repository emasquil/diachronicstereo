import torch.utils.data as data
import torch
import random
import numpy as np

from .middlebury_dataset import MiddleburyDataset
from .middlebury2021_dataset import Middlebury2021Dataset
from .kittistereo_dataset import KITTIStereoDataset
from .booster_dataset import BoosterDataset
from .layeredflow_dataset import LayeredFlowDataset
from .monotrap_dataset import MonoTrapDataset


def worker_init_fn(worker_id):
    torch_seed = torch.randint(0, 2**30, (1,)).item()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)


def fetch_dataloader(args):
    """Create the data loader for the corresponding trainign set"""

    dataset_test_dict = {
        "kitti_stereo": KITTIStereoDataset,
        "kitti2012": KITTIStereoDataset,
        "kitti2015": KITTIStereoDataset,
        "middlebury": MiddleburyDataset,
        "eth3d": MiddleburyDataset,
        "middlebury2021": Middlebury2021Dataset,
        "booster": BoosterDataset,
        "layeredflow": LayeredFlowDataset,
        "monotrap": MonoTrapDataset,
    }

    if args.dataset in dataset_test_dict:
        datapaths = args.datapath.split(";")
        if args.test:
            dataset = dataset_test_dict[args.dataset](
                datapaths[0], test=True, overfit=args.overfit, mono=None
            )
            for i in range(1, len(datapaths)):
                dataset += dataset_test_dict[args.dataset](
                    datapaths[i], test=True, overfit=args.overfit, mono=None
                )

            loader = data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                pin_memory=False,
                shuffle=False,
                num_workers=args.numworkers,
                drop_last=True,
            )
            print("Testing with %d image pairs" % len(dataset))
        else:
            raise NotImplementedError

    return loader
