import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    #filt[i] *= 255
    #filt[i] = filt[i].astype(np.uint8)


img = cv2.imread("imori.jpg")
H, W, C = img.shape
gray = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]

tmp = np.pad(gray, (K//2, K//2), 'edge')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)
fig, axes = plt.subplots(1, 4, subplot_kw={'xticks': [], 'yticks': []})
plt.gray()
for ax, i in zip(axes.flat, range(4)):
    out = np.zeros((H, W), dtype=np.float)
    for y in range(H):
        for x in range(W):
            #out[y, x] = np.sum(tmp[y:y+K, x:x+K] * filt[i]) / filt[i].sum()
            out[y, x] = np.sum(tmp[y:y+K, x:x+K] * filt[i])
    out[0>out] = 0
    out[255<out] = 255
    ax.imshow(out.astype(np.uint8))
    ax.set_title("Angle "+str(45*i))
plt.savefig("my_answer_79.png")

