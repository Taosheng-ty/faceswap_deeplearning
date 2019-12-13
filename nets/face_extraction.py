import os
import sys
import cv2
import imutils
import numpy as np
from PIL import Image


# Takes an image, detects any faces in the image, and returns a list of the 
# isolated face images. 
# (With help from Karan Bhanot: https://towardsdatascience.com/extracting-faces-using-opencv-face-detection-neural-network-475c5cd0c260)
#
# Parameters:
#   scene- a single png image file
#   model- the trained openCV facial recognition model
# Returns:
#   faces- list on images which is every isolated face from the scene
def face_extraction(scene, model):
  faces = []
  updated_images = []

  (h, w) = scene.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(scene, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

  model.setInput(blob)
  detections = model.forward()

  # Create frame around face
  for i in range(0, detections.shape[2]):
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    confidence = detections[0, 0, i, 2]

    # If confidence > 0.5, show box around face
    if (confidence > 0.5):
      cv2.rectangle(scene, (startX, startY), (endX, endY), (255, 255, 255), 2)

  updated_images.append(scene)

  # Identify each face
  count = 0
  for i in range(0, detections.shape[2]):
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    confidence = detections[0, 0, i, 2]
    # If confidence > 0.5, save it as a separate file
    if (confidence > 0.5):
      count += 1
      frame = scene[startY:endY, startX:endX]
      faces.append(frame)

  return faces
