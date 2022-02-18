import cv2
from PIL import Image
from pathlib import Path

route = f"{Path(__file__).parent}"

im = Image.open(route + "/tests/model_img.png").convert('RGB')
im.save(route+"/tests/model_img.jpg","jpeg")

