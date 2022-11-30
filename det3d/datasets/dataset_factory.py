from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset
from .kitti import KittiDataset

dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset,
    "KITTI": KittiDataset
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
