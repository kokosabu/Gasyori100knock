import cv2
import numpy as np

K = 11
s = 1.5
g = 1.2
l = 3
p = 0
A = 0

def x_dash(x, y):
    return np.cos(A*np.pi/180) * x + np.sin(A*np.pi/180) * y

def y_dash(x, y):
    return -np.sin(A*np.pi/180) * x + np.cos(A*np.pi/180) * y

def G(y, x):
    return np.exp(-(x_dash(x, y)**2 + g**2 * y_dash(x, y)**2) / (2 * s**2)) * np.cos(2 * np.pi * x_dash(x, y) / l + p)

filt = np.zeros((4, K, K))
for i in range(4):
    A = i * 45
    for y in range(K):
        for x in range(K):
            filt[i, y, x] = G(y-K//2, x-K//2)

    filt[i] /= np.sum(np.abs(filt[i]))

img = cv2.imread("imori.jpg")
H, W, C = img.shape
gray = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]

tmp = np.pad(gray, (K//2, K//2), 'edge')

out = np.zeros((H, W), dtype=np.float)
for i in range(4):
    fout = np.zeros((H, W), dtype=np.float)
    for y in range(H):
        for x in range(W):
            fout[y, x] = np.sum(tmp[y:y+K, x:x+K] * filt[i])
    fout[0>fout] = 0
    fout[255<fout] = 255
    out += fout

out = (out - np.min(out)) * (255 / (np.max(out) - np.min(out)))
out[0>out] = 0
out[255<out] = 255
cv2.imwrite("my_answer_80.jpg", out.astype(np.uint8))
