import cv2
import numpy as np

img = cv2.imread("imori.jpg")
H, W, C = img.shape
img = img.astype(np.float32)
r = img[:, :, 2] / 255
g = img[:, :, 1] / 255
b = img[:, :, 0] / 255

Max = np.zeros_like(r)
Min = np.zeros_like(r)
for i in range(len(Max)):
    for j in range(len(Max[i])):
        Max[i][j] = max(r[i][j], g[i][j], b[i][j])
        Min[i][j] = min(r[i][j], g[i][j], b[i][j])

h = np.zeros_like(r)
for i in range(len(h)):
    for j in range(len(h[i])):
        if Max[i][j] == Min[i][j]:
            h[i][j] = 0
        elif Min[i][j] == b[i][j]:
            h[i][j] = 60 * (g[i][j] - r[i][j]) / (Max[i][j] - Min[i][j]) + 60
        elif Min[i][j] == r[i][j]:
            h[i][j] = 60 * (b[i][j] - g[i][j]) / (Max[i][j] - Min[i][j]) + 180
        else:
            h[i][j] = 60 * (r[i][j] - b[i][j]) / (Max[i][j] - Min[i][j]) + 300

v = Max.copy()
s = Max.copy() - Min.copy()

out = np.zeros((H, W), dtype=np.uint8)
out[(180<=h) & (h<=260)] = 255

cv2.imwrite("my_answer_70.png", out)
