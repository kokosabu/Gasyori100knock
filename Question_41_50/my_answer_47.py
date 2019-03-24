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

img = cv2.imread("imori.jpg")
img = img.astype(np.float32)
H, W, C = img.shape

img = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]

max_Sb = 0
t = -1
for i in range(256):
    c0 = img[img < i]
    c1 = img[img >= i]
    w0 = c0.shape[0] / (img.shape[0]*img.shape[1])
    w1 = 1 - w0
    if c0.shape[0] != 0:
        M0 = np.average(c0)
    else:
        M0 = 0
    if c1.shape[0] != 0:
        M1 = np.average(c1)
    else:
        M1 = 0
    Sb = w0 * w1 * (M0 - M1)**2
    if Sb > max_Sb:
        max_Sb = Sb
        t = i

img = img.astype(np.uint8)
img = np.where(img < t, 0, 255)

out = dilation(img)
out = dilation(out)

cv2.imwrite("my_answer_47.jpg", out.astype(np.uint8))
