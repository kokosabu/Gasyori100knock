import cv2
import numpy as np

K_size = 3
pad = K_size // 2
Kx = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])
Ky = np.array([[ 1,  2,  1],
               [ 0,  0,  0],
               [-1, -2, -1]])

img = cv2.imread("thorino.jpg").astype(np.float)
H, W, C = img.shape

gray = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]

tmp = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
tmp[pad:pad+H, pad:pad+W] = gray

Ix = gray.copy()
Iy = gray.copy()
for y in range(H):
    for x in range(W):
        #Iy[y, x] = np.sum(Ky * tmp[y:y+K_size, x:x+K_size])
        #Ix[y, x] = np.sum(Kx * tmp[y:y+K_size, x:x+K_size])
        Iy[y, x] = np.mean(Ky * tmp[y:y+K_size, x:x+K_size])
        Ix[y, x] = np.mean(Kx * tmp[y:y+K_size, x:x+K_size])

IxIx = Ix**2
IyIy = Iy**2
IxIy = Ix*Iy
detH = IxIx*IyIy - IxIy**2

K = detH / (1 + IxIx + IyIy)**2

out = img.copy()
out[...,2] = gray
out[...,1] = gray
out[...,0] = gray
tmp = np.pad(detH, (pad, pad), 'edge')
for y in range(H):
    for x in range(W):
        #if np.max(tmp[y:y+K_size, x:x+K_size]) == detH[y,x] and (np.max(K) * 100000000) <= detH[y,x]:
        #if np.max(tmp[y:y+3, x:x+3]) == detH[y,x] and (np.max((IxIx[y,x], IyIy[y,x], IxIy[y,x])) * 0.1) <= detH[y,x]:
        if np.max(tmp[y:y+K_size, x:x+K_size]) == detH[y,x] and (np.max(detH) * 0.1) <= detH[y,x]:
            out[y, x] = (0, 0, 255)

cv2.imwrite("my_answer_81.jpg", out.astype(np.uint8))
