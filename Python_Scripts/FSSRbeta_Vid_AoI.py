## This file contains the script to run the FSSRbeta model on VIDEO
# pip install ultralytics  if you get an error on ultralytics
# Install Packages
from ultralytics import YOLO
import cv2
from hub_sdk import HUBClient

# Load the YOLOv8 pre-trained model
model = YOLO('/Users/jackorebaugh/Documents/Code/FSSR_ML/FSSRbeta.pt')

# Capture native webcam
cap = cv2.VideoCapture("/Users/jackorebaugh/Documents/Code/FSSR_ML/Data/Drone Flying Over Forest.mp4")

# While loop for capturing and running specified YOLO model on our webcam
while cap.isOpened():
    success, frame=cap.read()

    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): # Allowing for us to stop the video feed by pressing 'q'
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)