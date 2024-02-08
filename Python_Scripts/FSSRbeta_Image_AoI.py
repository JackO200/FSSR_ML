## This file contains the script to run the FSSRbeta model on IMAGES

# Install Packages
from ultralytics import YOLO
import cv2
from hub_sdk import HUBClient

# Install Packages
from ultralytics import YOLO
import cv2
from hub_sdk import HUBClient

# Load the YOLOv8 pre-trained model
model = YOLO('/Users/jackorebaugh/Documents/Code/FSSR_ML/FSSRbeta.pt')


# Run inference on test data
results = model(source="/Users/jackorebaugh/Documents/Code/FSSR_ML/Data/GoogleEarth_Test.png", show=True, save=True)

# Way to close the window
cv2.startWindowThread()
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)