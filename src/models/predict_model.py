
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import DatasetMapper, build_detection_test_loader

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

file_name = '30.jpg'

# Load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = '../../model_final.pth'

model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

dataloader = build_detection_test_loader([{'file_name': file_name}], mapper=DatasetMapper(cfg))
batch = next(iter(dataloader))

feats = get_object_features(model, batch)

