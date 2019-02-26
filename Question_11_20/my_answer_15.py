import cv2
import numpy as np

K_size = 3
pad = K_size // 2
Kv = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])
Kh = np.array([[ 1,  2,  1],
               [ 0,  0,  0],
               [-1, -2, -1]])

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

gray = 0.2126*img[:,:,2] + 0.7152*img[:,:,1] + 0.0722*img[:,:,0]
gray = gray.astype(np.uint8)

tmp = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
tmp[pad:pad+H, pad:pad+W] = gray

out_v = img.copy()
out_h = img.copy()
for y in range(H):
    for x in range(W):
        out_v[y, x] = np.sum(Kv * tmp[y:y+K_size, x:x+K_size])
        out_h[y, x] = np.sum(Kh * tmp[y:y+K_size, x:x+K_size])

out_v[out_v < 0] = 0
out_v[out_v > 255] = 255
out_h[out_h < 0] = 0
out_h[out_h > 255] = 255

cv2.imwrite("my_answer_15_v.jpg", out_v.astype(np.uint8))
cv2.imwrite("my_answer_15_h.jpg", out_h.astype(np.uint8))
