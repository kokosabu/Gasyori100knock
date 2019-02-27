import cv2
import numpy as np

K_size = 3
pad = K_size // 2

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape
gray = 0.2126*img[:,:,2] + 0.7152*img[:,:,1] + 0.0722*img[:,:,0]

K = np.array([[0,  1, 0],
              [1, -4, 1],
              [0,  1, 0]])

tmp = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
tmp[pad:H+pad, pad:W+pad] = gray

for y in range(H):
    for x in range(W):
        img[y, x] = np.sum(K * tmp[y:y+K_size, x:x+K_size])

img[img < 0] = 0
img[img > 255] = 255

cv2.imwrite("my_answer_17.jpg", img.astype(np.uint8))
