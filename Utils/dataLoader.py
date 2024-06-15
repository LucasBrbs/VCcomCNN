import os
import cv2
from sklearn.model_selection import train_test_split


#function to load images, resize to 100px, 100px and normalize between 0 and 1
def loadImagesFromFolder(imagePaths, imageReceiver):
    
    for path in os.listdir(imagePaths):
        path = os.path.join(imagePaths, path)
        # Verifica se o arquivo existe antes de tentar carregá-lo
        if os.path.exists(path):
            image = cv2.imread(path)
            if image is not None:
                resizedImage = cv2.resize(image,(100,100))
                normalizedImage = resizedImage / 255
                imageReceiver.append(normalizedImage)
            else:
                print(f"Error the image doenst found in path: {path}")
        else:
            print(f"Error the archive was missing in path: {path}")
    return imageReceiver

# Function to split the dataset into training, validation, and test sets

def splitDataset(images, test_size=0.2, validation_size=0.25, random_state=42):
    # Split into training+validation and test
    X_train_val, X_test = train_test_split(images, test_size=test_size, random_state=random_state)
    
    # Split training+validation into training and validation
    X_train, X_val = train_test_split(X_train_val, test_size=validation_size, random_state=random_state)
    
    return X_train, X_val, X_test