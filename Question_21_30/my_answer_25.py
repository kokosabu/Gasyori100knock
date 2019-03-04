import cv2
import numpy as np

scale = 1.5

img = cv2.imread("imori.jpg")
H, W, C = img.shape

out = np.zeros((int(H*scale), int(W*scale), C), dtype=np.uint8)

for y in range(int(H*scale)):
    for x in range(int(W*scale)):
        out[y, x] = img[round(y/scale), round(x/scale)]

cv2.imwrite("my_answer_25.jpg", out)
