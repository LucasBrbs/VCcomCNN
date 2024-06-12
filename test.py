import cv2
import tensorflow
import keras
import matplotlib as plt
import numpy as np
import pandas as pd
import sklearn as sk



# pip install opencv-python tensorflow keras matplotlib numpy pandas scikit-learn
#command to install all the libraries

# Read the image using OpenCV
image = cv2.imread('cells/ASC-H/0f023f6ad3d33b33334016798995fe53_377_402.tif')

# Show the image using OpenCV
cv2.imshow('Image', image)
cv2.waitKey(0)  # Press any key to close the window
