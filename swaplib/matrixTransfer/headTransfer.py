from ..faceboxloc.facebox import makeFaceBox
from ..stylelib.doStyle import doStyle
from ..seglib.faceSegmentation import face2parsing_maps, parsing2mask 
from cv2.ximgproc import guidedFilter

from PIL import Image

from .landmarks import *
from .swapLib import *
from pathlib import Path

route = f"{Path(__file__).parent}"
def show_landmarks(landmarks, image, fileName):
  print(f"Detected {len(landmarks)} faces. ")
  img_show = cv2.imread(image)
  for landmark in landmarks:
    for i,(x,y) in enumerate(landmark[0]):

      cv2.circle(img_show, (x,y),2,(255,255,255),2)
      img_show = cv2.putText(img_show, f"{i}", (int(x-5),int(y-5)), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.5, (255, 0, 0) , 1, cv2.LINE_AA)
  
  cv2.imwrite(fileName, img_show)

  # cv2_imshow(img_show)

def Swap(source_path, 
              # style_path, 
              ref_path, 
              headless_path ):
  
  eps = 5e-6
  eps *= 255*255
  COLOUR_CORRECT_BLUR_FRAC = 0.6
  LEFT_EYE_POINTS = list(range(42, 48))
  RIGHT_EYE_POINTS = list(range(36, 42))
  FACE_POINTS = list(range(17, 68))
  LEFT_EYE_POINTS = list(range(42, 48))
  RIGHT_EYE_POINTS = list(range(36, 42))
  LEFT_BROW_POINTS = list(range(22, 27))
  RIGHT_BROW_POINTS = list(range(17, 22))
  NOSE_POINTS = list(range(27, 35))
  MOUTH_POINTS = list(range(48, 61))
  JAW_POINTS_ = list(range(4, 13))
  ALIGN_POINTS = (JAW_POINTS_)

  OVERLAY_POINTS = [
      LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS+
      NOSE_POINTS + MOUTH_POINTS+JAW_POINTS_
  ]

  FEATHER_AMOUNT = 15 
  """
    source_path Path to the source face
    style_path deprecated
    ref_path reference template path
    headless_path headless template path
  """
  # Makefacebox
  face_box = makeFaceBox(source_path, save=False)
  cv2.imwrite("D:\Python_projects\headswap-machine/test.jpg",face_box)
  # Style transfer and colortransfer
  #face_stylized = doStyle(Image.fromarray(face_box))
  face_stylized = Image.fromarray(face_box)


  # Landmarks
  # possible problem? -> landmarks need to be of the facebox

  source_im = np.asarray(face_box)[...,::-1]
  style_im = np.asarray(face_stylized)[...,::-1]

  target_im = cv2.imread(ref_path, cv2.IMREAD_COLOR)
  head_less = cv2.imread(headless_path, cv2.IMREAD_COLOR)
  
  st_im = style_im[...,::-1]/255
  hl_tm = head_less[...,::-1]/255
  
  cv2.imwrite("D:\Python_projects\headswap-machine/test2.jpg",hl_tm)

  lm = landmarksFromImage([face_box[...,::-1], target_im[...,::-1]])
  source_lm = lm[0][0][0]
  target_lm = lm[1][0][0]



  #TODO[] ?fix this to do it inplace 
  parsing = face2parsing_maps(face_box)
  # masked_face, mask_face, points_face = parsing2mask(source_im, parsing, include=[0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]) 
  masked_face, mask_face = parsing2mask(source_im, parsing, include = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]) 
  mask_face = guidedFilter(source_im.astype(np.float32), mask_face.astype(np.float32), 40, eps)
  mask_face3d = np.array([mask_face,mask_face,mask_face]).transpose(1,2,0)
  # mask_face3d = get_masked_blur(mask_face, 15, (15,15)) #<- blurs and makes 3d mask
  # masked_hair, mask_hair, points_hair = parsing2mask(source_im, parsing, include=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0])  
  masked_hair, mask_hair = parsing2mask(source_im, parsing, include = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0])  

  

  # Hair mask process
  mask_hair_erode = cv2.erode(mask_hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 3)
  better_hair = guidedFilter(source_im.astype(np.float32), mask_hair_erode.astype(np.float32), 40, eps)
  better_hair3d = np.array([better_hair,better_hair,better_hair]).transpose(1,2,0)
  # mask_blur = get_masked_blur(mask, 15, (15,15)) 
  compose_mask =  better_hair3d + mask_face3d

  t_lm = np.matrix(target_lm[ALIGN_POINTS])
  s_lm = np.matrix(source_lm[ALIGN_POINTS])
  M = transformation_from_points(t_lm, s_lm) 
  warped_mask = warp_im(compose_mask, M, target_im.shape)


  warped_source_im = warp_im(style_im, M, target_im.shape)
    
  final = head_less*(1-warped_mask)+(warped_source_im*warped_mask)
 

   
  print("Swap Done")
  return final