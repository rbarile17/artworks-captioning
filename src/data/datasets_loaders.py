import os
from os import listdir

import copy
import json

import numpy as np
import torch
import pandas as pd

from detectron2.data import DatasetCatalog, DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from PIL import Image
from PIL.Image import DecompressionBombError

import urllib.request
from urllib.request import Request, HTTPError

import io

import time

import warnings

DETECTRON2_DATASETS = os.environ['DETECTRON2_DATASETS']

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

def coco_with_virtual_paintings(virtual_paintings_dir, train=True):
    if train:
        dataset_name = 'coco_2017_train'
    else:
        dataset_name = 'coco_2017_val'

    dataset = DatasetCatalog.get(dataset_name)
        
    virtual_paintings = [f for f in listdir(virtual_paintings_dir)]
    virtual_paintings.sort(key = lambda file: int(file.split('.')[0]))

    for image, virtual_painting in zip(dataset, virtual_paintings):
        image['file_basename'] = os.path.basename(os.path.normpath(image['file_name']))
        image['virtual_painting_file_name'] = virtual_painting
        image['virtual_painting_path'] = f"{virtual_paintings_dir}/{virtual_painting}"

    return dataset

def load_artgraph():
    artworks = listdir('../../data/datasets/artgraphv2/imagesf2/')
    dataset = [
        {"file_name": f"../../data/datasets/artgraphv2/imagesf2/{artwork}"} 
        for artwork in artworks
    ]

    return dataset

def load_artpedia():
    with open('../../data/datasets/artpedia/artpedia.json', 'r') as file:
        artpedia = json.load(file)

    return list(artpedia.values())

def load_semart():
    semart = pd.concat([
        pd.read_csv('../../data/datasets/semart/semart_train.csv', delimiter='\t', encoding='Cp1252'),
        pd.read_csv('../../data/datasets/semart/semart_val.csv', delimiter='\t', encoding='Cp1252'),
        pd.read_csv('../../data/datasets/semart/semart_test.csv', delimiter='\t', encoding='Cp1252')
    ], axis=0)

    semart = semart.rename(columns={'IMAGE_FILE': 'file_name', 'DESCRIPTION': 'description'})

    semart_dicts = semart.loc[:, ['file_name', 'description']].to_dict('records')

    for image in semart_dicts:
        image['file_name'] = f"../../data/datasets/semart/Images/{image['file_name']}"

    return semart_dicts

def load_ola():
    ola = pd.read_csv('../../data/datasets/ola/ola_dataset_release_v0.csv')
    ola = ola.rename(columns={'painting': 'file_name', 'utterance': 'description'})

    artgraph = os.listdir('../../data/datasets/artgraphv2/imagesf2/')

    ola = ola.loc[:, ['file_name', 'description']]

    # remove dupliates from ola keeping the first occurence
    ola = ola.drop_duplicates(subset=['file_name'], keep='first')

    ola = ola.to_dict('records')
    ola_dicts = []
    for image in ola:
        if f"{image['file_name']}.jpg" in artgraph:
            image['file_name'] = f"../../data/datasets/artgraphv2/imagesf2/{image['file_name']}.jpg"
            ola_dicts.append(image)

    return ola_dicts

datasets_loading_functions = {
    'ola': load_ola,
    'artgraph': load_artgraph,
    'artpedia': load_artpedia,
    'semart': load_semart,
}