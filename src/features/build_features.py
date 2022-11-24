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

from data.datasets_loaders import load_artgraph, load_artpedia, load_semart, load_ola
from data.datasets_loaders import URLMapper

import h5py

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

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

# param
cfg.MODEL.WEIGHTS = '../models/model_final_styleflow.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
model = build_model(cfg)

# param
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
DatasetCatalog.register("ola", load_ola)

dataset = DatasetCatalog.get("ola")
dataloader = build_detection_test_loader(dataset, mapper=DatasetMapper(cfg))

model.eval()
with torch.no_grad():
    with tqdm(desc=f'Progress', unit='iteration', total=len(dataloader)) as pbar:
        with h5py.File('../../data/features/ola/ola_detections_styleflow.hdf5', 'w') as h5_file:
            for batch in tqdm(dataloader):
                feats = get_object_features(model, batch)
                h5_features = h5_file.create_dataset(
                    f"{batch[0]['file_name'].split('/')[-1].split('.')[0]}_features", 
                    shape=feats.shape, 
                    dtype=np.float32)
                for j in range(0, feats.shape[0]):
                    h5_features[j] = feats[j].cpu()
                pbar.update()