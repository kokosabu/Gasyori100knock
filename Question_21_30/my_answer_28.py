import cv2
import numpy as np

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

out = np.zeros((H, W, C), dtype=np.float)

A = np.array([[1, 0,  30],
              [0, 1, -30],
              [0, 0,   1]])

for y in range(H):
    for x in range(W):
        (x_, y_, _) = np.matmul(A, np.array([x, y, 1]))
        if 0 <= y_ and y_ < H and 0 <= x_ and x_ < W:
            out[y_, x_] = img[y, x]

cv2.imwrite("my_answer_28.jpg", out.astype(np.uint8))

