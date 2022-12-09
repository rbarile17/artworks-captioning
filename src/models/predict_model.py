
import os
import pickle
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import DatasetMapper, build_detection_test_loader

# get environment variable meshed_memory_transformer path
meshed_memory_transformer_path = os.environ['MESHED_MEMORY_TRANSFORMER_PATH']

# import code from meshed-memory-transformer directory
import sys
sys.path.append(meshed_memory_transformer_path)
from models.transformer import (MemoryAugmentedEncoder, MeshedDecoder,
                                ScaledDotProductAttentionMemory, Transformer)
from data.field import TextField

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

def predict_single_image(model, image, text_field, device):
    model.eval()
    image = image.to(device)
    # unsqueeze to add batch dimension
    image = image.unsqueeze(0)
    with torch.no_grad():
        out, _ = model.beam_search(image, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
    caps_gen = text_field.decode(out, join_words=True)

    print(caps_gen)

file_name = './30.jpg'

# Load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = './model_final.pth'

object_detection_model = build_model(cfg)
DetectionCheckpointer(object_detection_model).load(cfg.MODEL.WEIGHTS)

dataloader = build_detection_test_loader([{'file_name': file_name}], mapper=DatasetMapper(cfg))
batch = next(iter(dataloader))

object_detection_model.eval()
with torch.no_grad():
    feats = get_object_features(object_detection_model, batch).cpu().numpy()

    delta = 50 - feats.shape[0]
    if delta > 0:
        precomp_data = np.concatenate([feats, np.zeros((delta, feats.shape[1]))], axis=0)
    elif delta < 0:
        precomp_data = feats[:50]

device = torch.device('cuda')

# Pipeline for text
text_field = TextField(
    init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
text_field.vocab = pickle.load(open('vocab_coco2017_train.pkl', 'rb'))

encoder = MemoryAugmentedEncoder(
    3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': 40})
decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
data = torch.load('./coco_2017_train_best.pth')
model.load_state_dict(data['state_dict'])

image = torch.from_numpy(feats).float()
predict_single_image(model, image, text_field, device)

# remove meshed-memory-transformer directory from path
sys.path.pop()
