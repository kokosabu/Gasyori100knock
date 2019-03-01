import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("imori_dark.jpg")
H, W, C = img.shape

a = 0
b = 255
c = np.min(img)
d = np.max(img)

img = img.astype(np.float)
out = img.copy()

for y in range(H):
    for x in range(W):
        for c_ in range(C):
            if img[y, x, c_] < c:
                out[y, x, c_] = a
            elif img[y, x, c_] <= d:
                out[y, x, c_] = (b-a)/(d-c) * (img[y, x, c_]-c) + a
            else:
                out[y, x, c_] = b

out = out.astype(np.uint8)
cv2.imwrite("my_answer_21_1.jpg", out)

plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("my_answer_21_2.png")
