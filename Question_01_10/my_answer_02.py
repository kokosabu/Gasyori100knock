import cv2
import numpy as np

img = cv2.imread("imori.jpg")
img2 = img.copy().astype(np.float32)

r = img2[:, :, 2]
g = img2[:, :, 1]
b = img2[:, :, 0]

img2 = 0.2126*r + 0.7152*g + 0.0722*b
cv2.imwrite("my_answer_02.jpg", img2.astype(np.uint8))
