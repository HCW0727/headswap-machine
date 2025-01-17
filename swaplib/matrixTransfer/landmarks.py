import cv2
import os
import numpy as np
import urllib.request as urlreq
from pathlib import Path

script_dir = Path(__file__).parent
haarcascade_path = f"{Path(__file__).parent}/models/haarcascade_frontalface_alt2.xml"
LBFmodel_path = f"{Path(__file__).parent}/models/lbfmodel.yaml"

def landmarksFromImage(loaded_imgs):
  """
    loaded_imgs are cv2.imread images in RGB form
    loaded_imgs -> [cv2.imread(img_path)[...,::-1]]
  """
  landmarks_arr = []
  detector = cv2.CascadeClassifier(haarcascade_path)
  landmark_detector = cv2.face.createFacemarkLBF()
  landmark_detector.loadModel(LBFmodel_path)

  for ld_img in loaded_imgs:
    image = ld_img
    image_landmarks = image.copy()
    image_draw = image.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(image_gray)
    _, landmarks = landmark_detector.fit(image_gray, faces)
    landmarks_arr.append(landmarks)

  return landmarks_arr

def landmarksFromPath(s_paths):
  images = [cv2.imread(img_path)[...,::-1] for img_path in s_paths]

  return landmarksFromPath(images)


