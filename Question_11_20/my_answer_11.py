import cv2
import numpy as np

K_size = 3

img = cv2.imread("imori.jpg")
img = img.astype(np.float)

H, W, C = img.shape
pad = K_size // 2

tmp = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float)
tmp[pad:pad+H, pad:pad+W] = img[:, :]

for y in range(H):
    for x in range(W):
        for c in range(C):
            img[y, x, c] = np.mean(tmp[y:y+K_size, x:x+K_size, c])

cv2.imwrite("my_answer_11.jpg", img.astype(np.uint8))
