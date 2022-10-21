import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class BSDSDataset(CustomDataset):
    """BSDS dataset.

    Args:
        split (str): Split txt file for BSDS.
    """

    CLASSES = ('background', 'edge')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, split, **kwargs):
        super(BSDSDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split, **kwargs
        )
        #print(self.img_dir)
        #print(self.split)
        assert osp.exists(self.img_dir) and self.split is not None
