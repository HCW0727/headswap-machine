from swaplib.matrixTransfer import headTransfer
import cv2
from pathlib import Path

route = f"{Path(__file__).parent}"


def main():
  result = headTransfer.Swap(
  source_path=route + "/tests/user_img(3).jpg", 
  ref_path=route + "/tests/model_img.jpg", 
  headless_path=route + "/tests/p4_model_img.jpg",
  )
  cv2.imwrite(route+"/tests/test_result.jpg", result)
if __name__ == "__main__":
  main()
