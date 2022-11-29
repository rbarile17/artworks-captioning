import os
import json
import pandas as pd

from os import listdir
from detectron2.data import DatasetCatalog

from . import ARTGRAPH_PATH, SEMART_PATH, ARTPEDIA_PATH, OLA_PATH  

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

# Style transfer target dataset

def load_artgraph():
    artgraph = pd.read_csv(ARTGRAPH_PATH / 'artgraph.csv')

    # add artgraph path to file_name
    artgraph['file_name'] = ARTGRAPH_PATH / 'images' / artgraph['file_name']

    # convert dataframe to list of dictionaries
    artgraph = artgraph.to_dict('records')

    return artgraph

# Evaluation datasets

def load_artpedia():
    with open(ARTPEDIA_PATH / '/artpedia.json', 'r') as file:
        artpedia = json.load(file)

    return list(artpedia.values())

def load_semart():
    semart = pd.concat([
        pd.read_csv(SEMART_PATH / '/semart_train.csv', delimiter='\t', encoding='Cp1252'),
        pd.read_csv(SEMART_PATH / '/semart_val.csv', delimiter='\t', encoding='Cp1252'),
        pd.read_csv(SEMART_PATH / '/semart_test.csv', delimiter='\t', encoding='Cp1252')
    ], axis=0)

    semart = semart.rename(columns={
        'IMAGE_FILE': 'file_name', 
        'DESCRIPTION': 'description',
        'TITLE': 'title'
    })

    # add semart path to file_name
    semart['file_name'] = SEMART_PATH / 'images' / semart['file_name']

    semart = semart.loc[:, ['file_name', 'description']].to_dict('records')

    return semart

def load_ola():
    ola = pd.read_csv(OLA_PATH / 'ola_filtered.csv')
    ola = ola.to_dict('records')

    # add artgraph path to file_name
    ola['file_name'] = ARTGRAPH_PATH / 'images' / ola['file_name']

# map dataset name to loading function
    
datasets_loading_functions = {
    # style transfer target dataset
    'artgraph': load_artgraph,

    # evaluation datasets
    'ola': load_ola,
    'artpedia': load_artpedia,
    'semart': load_semart,
}