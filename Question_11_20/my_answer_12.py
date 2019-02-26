import cv2
import numpy as np

K_size = 3

img = cv2.imread("imori.jpg")
img = img.astype(np.float)

pad = K_size // 2
H, W, C = img.shape

tmp = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float)
tmp[pad:H+pad, pad:W+pad] = img[:, :].astype(np.float)

kernel = np.array([[1/3,   0,   0],
                   [  0, 1/3,   0],
                   [  0,   0, 1/3]])

for y in range(H):
    for x in range(W):
        for c in range(C):
            img[y, x, c] = np.sum(tmp[y:y+K_size, x:x+K_size, c] * kernel)

cv2.imwrite("my_answer_12.jpg", img.astype(np.uint8))
