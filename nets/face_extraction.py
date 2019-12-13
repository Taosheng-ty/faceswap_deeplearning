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


## Example for how this function is called
## Load paths to access models, also retrieve image file of scene.
# base_dir = os.path.abspath('') + '/'
# prototxt_path = os.path.join(base_dir + 'deploy.prototxt')
# caffemodel_path = os.path.join(base_dir + 'weights.caffemodel')
# image_path = base_dir + 'image/image.png'

## Read the model
# model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

## Load image
# frame = cv2.imread(image_path)
# frame = imutils.resize(frame, width = 400, height = 200)

## Call facial_extractor with the given frame and trained models
# faces = facial_extractor(frame, model)

## Save faces to directory to make sure it worked correctly
# i = 0
# for face in faces:
  # cv2.imwrite(base_dir + 'faces/' + str(i) + '_' + file +'.png', face)
  # i += 1
