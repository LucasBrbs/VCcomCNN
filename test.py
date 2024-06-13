import cv2
import tensorflow
import keras
import matplotlib as plt
import numpy as np
import pandas as pd
import sklearn as sk
import os 



# pip install opencv-python tensorflow keras matplotlib numpy pandas scikit-learn
#command to install all the libraries

#Definiton of all images 
imagePatternSize = (100,100)

#function to load images 
def load_images(imagePaths):
    images = []
    for path in imagePaths:
        # Verifica se o arquivo existe antes de tentar carregá-lo
        if os.path.exists(path):
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
            else:
                print(f"Error the image doenst found in path: {path}")
        else:
            print(f"Error the archive was missing in path: {path}")
    return images

images = load_images("cells/ASC-H/")

for idx, img in enumerate(images):
    print(f"{idx + 1}")

# Read the image using OpenCV
#image = cv2.imread('cells/ASC-H/0f023f6ad3d33b33334016798995fe53_377_402.tif')

# Show the image using OpenCV
#cv2.imshow('Image', image)
cv2.waitKey(0)  # Press any key to close the window
