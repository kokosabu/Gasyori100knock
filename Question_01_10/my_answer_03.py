import cv2
import numpy as np

img = cv2.imread("imori.jpg")
img2 = img.copy().astype(np.float32)

r = img2[:, :, 2]
g = img2[:, :, 1]
b = img2[:, :, 0]

img2 = 0.2126*r + 0.7152*g + 0.0722*b
img2 = img2.astype(np.uint8)
img2 = np.where(img2 < 128, 0, 255)
# img2[out < 128] = 0
# img2[out >= 128] = 255
cv2.imwrite("my_answer_03.jpg", img2)
