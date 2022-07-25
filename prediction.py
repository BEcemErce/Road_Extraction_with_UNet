# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 22:40:51 2022

@author: Sade
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
def prepare_plot(origImage, origMask, predMask):
  # initialize our figure
  figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
  # plot the original image, its mask, and the predicted mask
  ax[0].grid(False)
  ax[1].grid(False)
  ax[2].grid(False)
  ax[0].imshow(origImage)
  ax[1].imshow(origMask)
  ax[2].imshow(predMask)
  # set the titles of the subplots
  ax[0].set_title("Image")
  ax[1].set_title("Original Mask")
  ax[2].set_title("Predicted Mask")
  # set the layout of the figure and display it
  figure.tight_layout()
  figure.show()

def make_predictions(model, imagePath, masked_path):
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
		orig = image.copy()
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
		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask)
  # load the image paths in our testing file and randomly select 10
