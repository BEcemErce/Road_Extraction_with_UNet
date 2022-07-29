# -*- coding: utf-8 -*-
"""

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
def prepare_plot(origImage, origMask, predMask):
  figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
  ax[0].grid(False)
  ax[1].grid(False)
  ax[2].grid(False)
  ax[0].imshow(origImage)
  ax[1].imshow(origMask)
  ax[2].imshow(predMask)
  ax[0].set_title("Image")
  ax[1].set_title("Original Mask")
  ax[2].set_title("Predicted Mask")
  figure.tight_layout()
  figure.show()

def make_predictions(model, imagePath, masked_path):
	# set model to evaluation mode
	model.eval()
	with torch.no_grad():		
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		image = cv2.resize(image, (512, 512))
		orig = image.copy()
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

		prepare_plot(orig, gtMask, predMask)

