import cv2
import numpy as np

img1 = cv2.imread("imori.jpg").astype(np.float)
img2 = cv2.imread("thorino.jpg").astype(np.float)

alpha = 0.6
out = img1 * alpha + img2 * (1-alpha)

cv2.imwrite("my_answer_60.jpg", out.astype(np.uint8))
