import cv2
from pathlib import Path

route = f"{Path(__file__).parent}"

src = cv2.imread(route + '/tests/p4_model_img.jpg')
dst = cv2.resize(src,(712,1066),interpolation=cv2.INTER_LINEAR)


cv2.imwrite(route + '/tests/p4_model_img.jpg',dst)