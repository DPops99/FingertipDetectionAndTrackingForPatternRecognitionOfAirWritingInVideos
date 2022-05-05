import cv2
import numpy as np


if __name__ == "__main__":
    img_path = "/home/popa/Documents/fingerDetectionAndTracking/hgr1_skin/skin_masks/1_P_hgr1_id01_1.bmp"
    img = cv2.imread(img_path)
    cv2.imshow(img)