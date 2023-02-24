# Copyright (c) OpenMMLab. All rights reserved.
import collections
from collections.abc import Sequence
from copy import deepcopy

from mmcv.utils import build_from_cfg
from scipy.ndimage import gaussian_filter
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

@PIPELINES.register_module()
class MaskCompose(object):
    def __init__(self, transforms, prob, lambda_limits=(4, 16), keep_original=False):
        self.keep_original = keep_original
        self.prob = prob
        assert 0.0 <= self.prob <= 1.0

        assert isinstance(lambda_limits, Sequence)
        assert len(lambda_limits) == 2
        assert 0.0 < lambda_limits[0] < lambda_limits[1]
        self.lambda_limits = lambda_limits

        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f"transform must be callable or a dict, but got {type(transform)}")

    @staticmethod
    def _apply_transforms(data, transforms):
        for t in transforms:
            data = t(data)
            if data is None:
                return None

        return data

    @staticmethod
    def _generate_mask(shape, lambda_limits):
        noise = np.random.randn(*shape)

        sigma = np.exp(np.log10(np.random.uniform(lambda_limits[0], lambda_limits[1])))
        soft_mask = gaussian_filter(noise, sigma=sigma)

        threshold = np.median(soft_mask)
        hard_mask = soft_mask > threshold

        return hard_mask

    @staticmethod
    def _mix_img(main_img, aux_img, mask):
        return np.where(np.expand_dims(mask, axis=2), main_img, aux_img)

    def __call__(self, data):
        main_data = self._apply_transforms(deepcopy(data), self.transforms)
        assert main_data is not None
        if not self.keep_original and np.random.rand() > self.prob:
            return main_data

        aux_data = self._apply_transforms(deepcopy(data), self.transforms)
        assert aux_data is not None

        assert main_data["img"].shape == aux_data["img"].shape

        mask = self._generate_mask(main_data["img"].shape[:2], self.lambda_limits)
        mixed_img = self._mix_img(main_data["img"], aux_data["img"], mask)

        if self.keep_original:
            main_data["aux_img"] = mixed_img
        else:
            main_data["img"] = mixed_img

        return main_data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"

        return format_string
