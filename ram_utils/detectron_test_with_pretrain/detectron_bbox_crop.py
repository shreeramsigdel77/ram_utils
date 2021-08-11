# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def show_img(img,windowName:str)-> None:
    cv2.imshow(windowName,img)
    
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyWindow(windowName)    

def read_img(path:str)-> np.ndarray:
    img = cv2.imread(path)
    return img

def draw_bbox(image:np.ndarray, bbox_list:list,preview:bool=False)-> np.ndarray: 
    img = cv2.rectangle(image,(int(bbox_list[0]),int(bbox_list[1])),(int(bbox_list[2]),int(bbox_list[3])),(255,0,0),2)
    if preview:
        show_img(img,"bounding box")
    return img

def crop_from_bbox(image:np.ndarray,bbox_list:list,preview:bool=False):
    cropped_image = image[int(bbox_list[1]):int(bbox_list[3]), int(bbox_list[0]):int(bbox_list[2])]
    # print(cropped_image.shape)
    if preview:
        show_img(cropped_image,"cropedWindow")
    # pass



#read img
im = read_img("./input.jpg")
#show
# show_img(im,"Input 1")

cfg = get_cfg()


cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
import numpy
class_list = outputs["instances"].pred_classes.to("cpu").numpy()
bboxes = outputs["instances"].pred_boxes
print(class_list)
print(bboxes)

bbox_list = []
for i in bboxes.__iter__():
    # print(i.cpu().numpy())
    bbox_list.append(i.cpu().numpy())
im_draw_bbox_copy = im.copy()
for lable, bbox in zip (class_list,bbox_list):
    im_copy = im.copy()
    if lable == 0:    #label 0 is person class
        # print(lable)
        # print(bbox)
        draw_bbox(im_draw_bbox_copy,bbox,True)
        crop_from_bbox(im_copy,bbox,True)

#  use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2_imshow(out.get_image()[:, :, ::-1])
im_out = out.get_image()[:, :, ::-1]
#show
show_img(im_out,"Input 2")