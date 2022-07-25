# Road_Extraction_with_UNet


## Overview
Road extraction has an important role in many areas such as traffic management, urban planning, automatic vehicle navigation, emergency management, etc. Especially in developing countries, in disaster zones, maps and accessibility information are crucial. So, the main topic of this project is automatically extracting roads and street networks from satellite images. For achieving this purpose, the image segmentation process which is one of the computer vision subfields will apply.

## Goals
•	Extraction of road maps from satellite images  <br />
•	Applying the U-Net method with a success
 
## Methods & Data
Image segmentation is a method that separates the images into subgroups. By doing this, images are changed to more meaningful and easier to analyze representations. There are some subfields of image segmentation. In this project, one of these subfields which are semantic segmentation will discuss. As a method, a variant of deep Convolutional Neural Networks (CNN) which is known as U-Net will be used. U-Net is an encoder-decoder type network architecture for image segmentation. The name of the architecture comes from its unique shape. The architecture of the  original U-Net  is shown in the figure.

 
![image](https://user-images.githubusercontent.com/66211576/180845984-3d860956-9302-4df7-80c8-121b9f36d916.png)


The data properties: <br />
·	The training data for Road Challenge contains 6226 satellite imagery in RGB, size 1024x1024, and 1101 test images (but no masks) <br />
·	The imagery has a 50cm pixel resolution, collected by DigitalGlobe's satellite. <br />
·	Each satellite image is paired with a mask image for road labels. The mask is a grayscale image, with white standing for the road pixel, and black standing for the background. <br />
·	File names for satellite images and the corresponding mask image are id _sat.jpg and id _mask.png. id is a randomized integer. <br />

Data URL: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset

## Implementation

In the original U-Net architecture there are 4 convolution blocks in the each decoder and encoder parts. In this study, lots of architecture structures including original U-Net were applied and then the best arhitecture was selected which has the 3 convolution blocks in the encoder and decoder. You can find an implementation example of these modules in the implementation.ipynb. The model was trained with learning rate 2e-4 and batch size 8. The IOU score of this model was 0.43. The masks that was predicted by the model are shown in below

![image](https://user-images.githubusercontent.com/66211576/180850884-4727faeb-21e9-4345-b41e-5feac8ff1050.png)


## References
https://pyimagesearch.com/ <br />
Minaee, S., Boykov, Y. Y., Porikli, F., Plaza, A. J., Kehtarnavaz, N., & Terzopoulos, D. (2021). 
“Image segmentation using deep learning: A survey”. IEEE transactions on pattern analysis and machine 
intelligence <br />
Shelhamer E, Long J, Darrell T (April 2017). "Fully Convolutional Networks for Semantic 
Segmentation". IEEE Transactions on Pattern Analysis and Machine Intelligence. 39 (4): 640– 651. 
arXiv:1411.4038  <br />
Ronneberger O, Fischer P, Brox T (2015). "U-Net: Convolutional Networks for Biomedical Image 
Segmentation". arXiv:1505.04597. <br />
Zhang, Z., Liu, Q., & Wang, Y. (2018). Road extraction by deep residual u-net. IEEE Geoscience 
and Remote Sensing Letters, 15(5), 749-753. (2018). Road extraction by deep residual u-net. IEEE 
Geoscience and Remote Sensing Letters, 15(5), 749-753. <br />
Piao S, Jiaming L (2019)."Accuracy improvement of UNet based on dilated convolution." In Journal 
of Physics: Conference Series, vol. 1345, no. 5, p. 052066. IOP Publishing


