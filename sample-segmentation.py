#Segmenting the south africa death records

# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
import subprocess
from shlex import quote
import os
import sys
from pathlib import Path

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from detectron2.data.datasets import register_coco_instances
register_coco_instances("train", {}, "./training_set/sa_train_coco.json", "")

cfg = get_cfg()
cfg.merge_from_file("./mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.MAX_ITER = 1000    # 1000 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 500   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Cause of Death

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
#trainer.train()

img_dir = sys.argv[1]

cfg.DATASETS.TEST = ("train", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
predictor = DefaultPredictor(cfg)



img_paths = list(Path(img_dir).rglob("*.jpg"))
print(img_paths)
for im in img_paths:
    image_path = str(im)
    image_array = cv2.imread(image_path)

    outputs = predictor(image_array)
    name = outputs["instances"].pred_classes
    classes = outputs["instances"].pred_boxes

    #0=Cause of Death
    rows = classes.tensor.cpu().numpy()
    row_name = name.cpu().numpy()

    print(im)
    print(row_name)

    j = 0
    for row in rows:
        left = int(row[0])
        top = int(row[1])
        right = int(row[2])
        bottom = int(row[3])
        cropped_array = image_array[top:bottom,left:right]

        if row_name[j] == 0:
            print('snippets/'+ image_path[5:-4] + '_cod.jpg')
            cv2.imwrite('snippets/'+ image_path[5:-4] + '_cod.jpg', cropped_array)
        j += 1
