import cv2
import numpy as np

K_size = 3

img = cv2.imread("imori.jpg").astype(np.float)

pad = K_size // 2
H, W, C = img.shape

r = img[:, :, 2]
g = img[:, :, 1]
b = img[:, :, 0]
gray = 0.2126*r + 0.7152*g + 0.0722*b
gray = gray.astype(np.uint8)

tmp = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
tmp[pad:H+pad, pad:W+pad] = gray[:, :].astype(np.float)

for y in range(H):
    for x in range(W):
        img[y, x] = np.max(tmp[y:y+K_size, x:x+K_size]) - np.min(tmp[y:y+K_size, x:x+K_size])

cv2.imwrite("my_answer_13.jpg", img.astype(np.uint8))
