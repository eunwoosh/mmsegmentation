# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class KvasirSegDataset(CustomDataset):
    """KvasirSeg Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('background', 'target')

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
