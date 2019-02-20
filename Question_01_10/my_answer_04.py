import cv2
import numpy as np

img = cv2.imread("imori.jpg")
img = img.astype(np.float32)

r = img[:, :, 2]
g = img[:, :, 1]
b = img[:, :, 0]

img = 0.2126*r + 0.7152*g + 0.0722*b

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

cv2.imwrite("my_answer_04.jpg", img)
