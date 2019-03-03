import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

Zmax = 255
S = H*W*C
Sum = np.sum(img)

out = img.copy()
Sum = 0
for i in range(256):
    Sum += np.sum(img == i)
    out[img == i] = Zmax / S * Sum

img = out.astype(np.uint8)

cv2.imwrite("my_answer_23_1.jpg", img)

plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("my_answer_23_2.png")
