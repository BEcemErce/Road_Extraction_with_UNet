# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import torch, torchvision
import cv2
import os

SMOOTH = 1e-6
def iou_numpy(outputs: np.array, labels: np.array):    
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)    
    return iou 

def IOU(model, imagePath,masked_path):
  model.eval()
  with torch.no_grad():
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
   
    image = cv2.resize(image, (512, 512))

    filename = imagePath.split(os.path.sep)[-1]
    filename=filename.replace("_sat.jpg","_mask.png")

    groundTruthPath = os.path.join(masked_path,filename)

    gtMask = cv2.imread(groundTruthPath,0)
    gtMask = cv2.resize(gtMask, (512,512),interpolation=cv2.INTER_AREA)

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).to("cuda")

    predMask = model(image).squeeze()
    predMask = torch.sigmoid(predMask)
    predMask = predMask.cpu().numpy()

    predMask = (predMask > 0.5) * 255
    predMask = predMask.astype(np.uint8)

    IOU=iou_numpy(predMask,gtMask)
    return(IOU)
    
