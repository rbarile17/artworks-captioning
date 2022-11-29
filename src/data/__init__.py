import os
from pathlib import Path

DETECTRON2_DATASETS = Path(os.environ['DETECTRON2_DATASETS'])

DATA_PATH = Path('./data')
RAW_DATA_PATH = DATA_PATH / 'raw'

ARTGRAPH_PATH = RAW_DATA_PATH / 'artgraph'
SEMART_PATH = RAW_DATA_PATH / 'semart'
ARTPEDIA_PATH = RAW_DATA_PATH / 'artpedia'