import cv2
import numpy as np

def dilation(img):
    K_size = 3
    pad = 1
    K = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]])

    H, W = img.shape
    in_img = np.pad(img, (pad, pad), 'edge')
    out = np.zeros((H, W), dtype=np.float)
    for y in range(H):
        for x in range(W):
            out[y, x] = in_img[y+pad, x+pad]
            if in_img[y+pad, x+pad] != 0:
                continue
            if np.sum(in_img[y:y+K_size, x:x+K_size]*K) >= 255:
                out[y, x] = 255

    return out

def erosion(img):
    K_size = 3
    pad = 1
    K = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]])

    H, W = img.shape
    in_img = np.pad(img, (pad, pad), 'edge')
    out = np.zeros((H, W), dtype=np.float)
    for y in range(H):
        for x in range(W):
            out[y, x] = in_img[y+pad, x+pad]
            if in_img[y+pad, x+pad] != 255:
                continue
            if np.sum(in_img[y:y+K_size, x:x+K_size]*K) < (255*4):
                out[y, x] = 0

    return out

def opening(img, N):
    out = img.copy()
    for i in range(N):
        out = erosion(out)
    for i in range(N):
        out = dilation(out)
    return out

def closing(img, N):
    out = img.copy()
    for i in range(N):
        out = dilation(out)
    for i in range(N):
        out = erosion(out)
    return out

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
mask = mask * 255
mask = closing(mask, 5)
mask = opening(mask, 5)
cv2.imwrite("my_answer_72_mask.png", mask)
mask = 1 - (mask / 255)

out = img.copy()
out[...,0] *= mask
out[...,1] *= mask
out[...,2] *= mask

cv2.imwrite("my_answer_72.jpg", out)
