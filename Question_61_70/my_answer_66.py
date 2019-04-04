import cv2
import numpy as np

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

gray = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]
gray2 = np.pad(gray, (1, 1), 'edge')

gx = gray2[1:H+1, 2:W+2] - gray2[1:H+1, 0:W]
gx[gx == 0] = 0.000001
gy = gray2[2:H+2, 1:W+1] - gray2[0:H, 2:W+2]

mag = np.sqrt(gx ** 2 + gy ** 2)
ang = np.arctan(gy / gx) * (180/np.pi)
ang[ang < 0] += 360
ang[ang > 180] -= 180

for i in range(9):
    ang[np.where((ang >= i*20) & (ang < (i+1)*20), True, False)] = i
ang[ang == 180] = 8


d = np.max(mag) - np.min(mag)
mag2 = np.round((mag - np.min(mag)) * (255 / d))

out = np.zeros((H, W, 3), dtype=np.uint8)
C = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
     [127, 127, 0], [127, 0, 127], [0, 127, 127]]
for i in range(9):
    out[ang == i] = C[i]

cv2.imwrite("my_answer_66_mag.jpg", mag2.astype(np.uint8))
cv2.imwrite("my_answer_66_gra.jpg", out.astype(np.uint8))
