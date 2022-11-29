import numpy as np
from tqdm import tqdm
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import DatasetCatalog
from detectron2.data import DatasetMapper, build_detection_test_loader

import h5py
import os
from ..data.datasets_loaders import datasets_loading_functions
from ..data import PROCESSED_DATA_PATH
from .. import load_params

def get_object_features(model, batch):
    images = model.preprocess_image(batch)  # don't forget to preprocess
    features = model.backbone(images.tensor)  # set of cnn features
    
    proposals, _ = model.proposal_generator(images, features, None)  # RPN outputs boxes

    features_ = [features[f] for f in model.roi_heads.box_in_features] # arrange features as a list
    box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates

    predictions = model.roi_heads.box_predictor(box_features)
    
    _, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)

    feats = box_features[pred_inds]

    return feats

def main(params=None):
    if params is None:
        params = load_params()

    # Load model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{params['model_name']}.yaml"))

    if params['is_pretrained']:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{params['weights']}.yaml")
    else:
        cfg.MODEL.WEIGHTS = params['weights']

    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # Load dataset
    if params['dataset'] not in ['coco_2017_train', 'coco_2017_val', 'coco_2017_test']:
        DatasetCatalog.register(params['dataset'], datasets_loading_functions[params['dataset']])

    dataset = DatasetCatalog.get(params['dataset'])
    dataloader = build_detection_test_loader(dataset, mapper=DatasetMapper(cfg))

    # Build features
    model.eval()
    with torch.no_grad():
        os.makedirs(PROCESSED_DATA_PATH / params['dataset'], exist_ok=True)
        with tqdm(desc=f'Progress', unit='iteration', total=len(dataloader)) as pbar:
            with h5py.File(
                f"{PROCESSED_DATA_PATH}/{params['dataset']}\
                    /{params['dataset']}_detections_{params['model_name']}.hdf5", 
                'w'
            ) as h5_file:
                
                for batch in tqdm(dataloader):
                    feats = get_object_features(model, batch)
                    h5_features = h5_file.create_dataset(
                        f"{batch[0]['file_name'].split('/')[-1].split('.')[0]}_features", 
                        shape=feats.shape, 
                        dtype=np.float32)
                    for j in range(0, feats.shape[0]):
                        h5_features[j] = feats[j].cpu()
                    pbar.update()

if __name__ == "__main__":
    main()