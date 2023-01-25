import io
import time
import warnings
import copy

import torch
import numpy as np

import urllib.request

from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from PIL import Image
from PIL.Image import DecompressionBombError

from urllib.request import Request, HTTPError


class CustomMapper(DatasetMapper):
    def __init__(self, cfg, is_train):
        params = super().from_config(cfg, is_train=is_train)
        super().__init__(**params)
    
    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)

        virtual_painting = utils.read_image(
            dataset_dict["virtual_painting_path"], 
            format=self.image_format)

        augs = T.AugmentationList([T.Resize(dataset_dict['image'].shape[1:])])

        aug_input = T.AugInput(virtual_painting)
        augs(aug_input)
        virtual_painting = aug_input.image

        dataset_dict['virtual_painting'] = torch.as_tensor(
            np.ascontiguousarray(virtual_painting.transpose(2, 0, 1)))

        return dataset_dict

class URLMapper(DatasetMapper):
    def __init__(self, cfg, is_train):
        params = super().from_config(cfg, is_train=is_train)
        params['augmentations'] = []
        super().__init__(**params)
        warnings.simplefilter("error", category=Image.DecompressionBombWarning)

    
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        request = Request(dataset_dict['img_url'])
        request.add_header(
            'User-Agent', 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36')
        try:
            with urllib.request.urlopen(request) as url:
                f = io.BytesIO(url.read())
        except HTTPError as err:
            if err.code == 404:
                raise err
            elif err.code == 429:
                time.sleep(1)
                with urllib.request.urlopen(request) as url:
                    f = io.BytesIO(url.read())
            else:
                raise

        try:
            image = Image.open(f)
        except Image.DecompressionBombError as e:
            url = dataset_dict['img_url']
            raise DecompressionBombError(url) from e
        except Image.DecompressionBombWarning as e:
            url = dataset_dict['img_url']
            raise DecompressionBombError(url) from e
        image = utils._apply_exif_orientation(image)
        image = utils.convert_PIL_to_numpy(image, self.image_format)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        return dataset_dict 