import os
from pathlib import Path

DETECTRON2_DATASETS = Path(os.environ['DETECTRON2_DATASETS'])

DATA_PATH = Path('./data')
RAW_DATA_PATH = DATA_PATH / 'raw'
PROCESSED_DATA_PATH = DATA_PATH / 'processed'

ARTGRAPH_PATH = RAW_DATA_PATH / 'artgraph'
SEMART_PATH = RAW_DATA_PATH / 'semart'
ARTPEDIA_PATH = RAW_DATA_PATH / 'artpedia'
OLA_PATH = RAW_DATA_PATH / 'ola'
COCO_TRAIN2017 = RAW_DATA_PATH / 'coco/train2017'

dataset_paths = {
    'artgraph': ARTGRAPH_PATH,
    'artgraph_images': ARTGRAPH_PATH / 'images',
    'artgraph_filter': ARTGRAPH_PATH / 'filter.csv',
    'semart': SEMART_PATH,
    'artpedia': ARTPEDIA_PATH,
    'ola': OLA_PATH,
    'coco_train2017': COCO_TRAIN2017,
}