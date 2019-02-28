import cv2
import numpy as np

def LoG(x, y, s):
    return (x**2 + y**2 - s**2) / (2 * np.pi * s**6) * np.exp(-(x**2+y**2) / (2*s**2))

K_size = 5
pad = K_size // 2

img = cv2.imread("imori_noise.jpg").astype(np.float)
H, W, C = img.shape

gray = 0.2126*img[:,:,2] + 0.7152*img[:,:,1] + 0.0722*img[:,:,0]
tmp = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
tmp[pad:pad+H, pad:pad+W] = gray

K = np.zeros((K_size, K_size), dtype=np.float)
for y in range(-pad, K_size-pad):
    for x in range(-pad, K_size-pad):
        K[y+pad, x+pad] = LoG(x, y, 3)
K /= K.sum()

for y in range(H):
    for x in range(W):
        img[y, x] = np.sum(K * tmp[y:y+K_size, x:x+K_size])

img[img < 0] = 0
img[img > 255] = 255

cv2.imwrite("my_answer_19.jpg", img.astype(np.uint8))
