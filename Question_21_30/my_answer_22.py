import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("imori_dark.jpg").astype(np.float)
H, W, C = img.shape

m0 = 128
s0 = 52

m = np.mean(img)
s = np.std(img)

img = s0 / s * (img - m) + m0

img[img < 0] = 0
img[img > 255] = 255
img = img.astype(np.uint8)

cv2.imwrite("my_answer_22_1.jpg", img)

plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("my_answer_22_2.png")
