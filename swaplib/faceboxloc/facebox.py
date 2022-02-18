# Function that executes the code (this is one import all the rest)
import cv2
from cv2 import bitwise_and,bitwise_not,bitwise_or
from .facelib import *
from ..seglib.faceSegmentation import *

route = f"{Path(__file__).parent}"

def makeFaceBox(source_path, cropface_pathsave='./crop_source.jpg', save=True):
  """
    Execution of the face box localization
    accepts an cv2 numpy image and returns or saves the cropped image
  """

  source_im = cv2.imread(source_path)
  masked, seg_mask = parsing2mask(source_im)
  cropped_face = crop_em(source_im, seg_mask)
  
  box = cv2.boundingRect(seg_mask.astype(np.uint8))
  
  seg_mask = cropface(seg_mask,box,fill=50)
  
  cv2.imshow("seg_mask1",seg_mask)
    
  _,seg_mask = cv2.threshold(seg_mask,0,255,cv2.THRESH_BINARY)
    
  cv2.imwrite(route+'seg_mask.jpg',seg_mask) 
  seg_mask = cv2.imread(route+'seg_mask.jpg') 
    
  print(cropped_face.shape,seg_mask.shape)
  
  cv2.imshow("cropped_face",cropped_face)
  cv2.imshow("seg_mask",seg_mask)
  
  inv_seg_mask = bitwise_not(seg_mask)
  
  testimg = bitwise_and(cropped_face,seg_mask)
  
  output = bitwise_or(testimg,inv_seg_mask)
  
  cv2.imshow("test",output)
  cv2.waitKey(0)
  
  return output[...,::-1]

#makeFaceBox("D:\Python_projects\headswap-machine\tests/user_img(3).jpg",save = False)
  