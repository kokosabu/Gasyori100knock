import cv2
import numpy as np

def g(x, y, s):
    return 1 / (s*np.sqrt(2 * np.pi)) * np.exp( - (x**2 + y**2) / (2*s**2))

img = cv2.imread("imori_noise.jpg")
img = img.astype(np.float)

H, W, C = img.shape
tmp = np.zeros((H+2, W+2, C), dtype=np.float)

tmp[1:-1, 1:-1] = img[:, :]

#kernel = np.array([[1, 2, 1],
#                   [2, 4, 2],
#                   [1, 2, 1]])

kernel = np.zeros((3, 3), dtype=np.float)
for x in range(3):
    for y in range(3):
        kernel[y, x] = g(x-1, y-1, 1.3)
kernel /= kernel.sum()

for i in range(H):
    for j in range(W):
        for c in range(C):
            #img[i, j, c] = np.sum(tmp[i:i+3, j:j+3, c] * kernel) / 16
            img[i, j, c] = np.sum(tmp[i:i+3, j:j+3, c] * kernel)

cv2.imwrite("my_answer_09.jpg", img.astype(np.uint8))
