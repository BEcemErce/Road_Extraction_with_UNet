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
  # set model to evaluation mode
  model.eval()
  # turn off gradient tracking
  
  with torch.no_grad():
    # load the image from disk, swap its color channels, cast it
    # to float data type, and scale its pixel values

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    # resize the image and make a copy of it for visualization
    image = cv2.resize(image, (512, 512))
    # find the filename and generate the path to ground truth
    # mask
    filename = imagePath.split(os.path.sep)[-1]
    filename=filename.replace("_sat.jpg","_mask.png")

    groundTruthPath = os.path.join(masked_path,filename)
    # load the ground-truth segmentation mask in grayscale mode
    # and resize it
    gtMask = cv2.imread(groundTruthPath,0)
    gtMask = cv2.resize(gtMask, (512,512),interpolation=cv2.INTER_AREA)
    # make the channel axis to be the leading one, add a batch
    # dimension, create a PyTorch tensor, and flash it to the
    # current device
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).to("cuda")
    # make the prediction, pass the results through the sigmoid
    # function, and convert the result to a NumPy array
    predMask = model(image).squeeze()
    predMask = torch.sigmoid(predMask)
    predMask = predMask.cpu().numpy()
    # filter out the weak predictions and convert them to integers
    predMask = (predMask > 0.5) * 255
    predMask = predMask.astype(np.uint8)

    IOU=iou_numpy(predMask,gtMask)
    return(IOU)
    