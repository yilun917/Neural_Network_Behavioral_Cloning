import numpy as np
import cv2
image = cv2.imread("./center_2016_12_01_13_42_07_892.jpg")
image_flipped = np.fliplr(image)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./flipped_center_2016_12_01_13_42_07_892.jpg',image_flipped)
cv2.imwrite('./gray_center_2016_12_01_13_42_07_892.jpg',image_gray)