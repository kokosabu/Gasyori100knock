import cv2
import numpy as np

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape
r = img[:, :, 2] / 255
g = img[:, :, 1] / 255
b = img[:, :, 0] / 255

Max = np.zeros_like(r)
Min = np.zeros_like(r)
for i in range(H):
    for j in range(W):
        Max[i][j] = max(r[i][j], g[i][j], b[i][j])
        Min[i][j] = min(r[i][j], g[i][j], b[i][j])

h = np.zeros_like(r)
h[Min == b] = 60 * (g[Min==b] - r[Min==b]) / (Max[Min==b] - Min[Min==b]) + 60
h[Min == r] = 60 * (b[Min==r] - g[Min==r]) / (Max[Min==r] - Min[Min==r]) + 180
h[Min == g] = 60 * (r[Min==g] - b[Min==g]) / (Max[Min==g] - Min[Min==g]) + 300
v = Max.copy()
s = Max.copy() - Min.copy()

mask = np.zeros((H, W), dtype=np.uint8)
mask[(180<=h) & (h<=260)] = 1
mask = 1 - mask

out = img.copy()
out[...,0] *= mask
out[...,1] *= mask
out[...,2] *= mask
#out = img * mask

cv2.imwrite("my_answer_71.jpg", out)
