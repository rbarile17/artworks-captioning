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

dataset_paths = {
    'artgraph': ARTGRAPH_PATH,
    'semart': SEMART_PATH,
    'artpedia': ARTPEDIA_PATH,
    'ola': OLA_PATH
}