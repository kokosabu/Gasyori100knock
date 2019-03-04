import cv2
import numpy as np

img = cv2.imread("imori_gamma.jpg").astype(np.float)

c = 1
g = 2.2

img /= 255
img = (1/c * img) ** (1/g)
img *= 255
img[img < 0] = 0
img[img > 255] = 255
img = img.astype(np.uint8)

cv2.imwrite("my_answer_24.jpg", img)
