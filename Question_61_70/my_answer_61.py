import cv2
import numpy as np

img = cv2.imread("renketsu.png").astype(np.float)
H, W, C = img.shape

tmp = np.zeros((H, W), dtype=np.uint8)
tmp[np.sum(img, axis=2) != 0] = 1

tmp2 = np.pad(tmp, (1, 1), 'edge')

S = np.full((H, W), 5, dtype=np.float)

for j in range(H):
    for i in range(W):
        if tmp[j, i] == 0:
            continue
        x = tmp2[j:j+3, i:i+3]
        S[j, i] = (x[1,2] - (x[1,2]*x[0,2]*x[0,1])) +\
                  (x[0,1] - (x[0,1]*x[0,0]*x[1,0])) +\
                  (x[1,0] - (x[1,0]*x[2,0]*x[2,1])) +\
                  (x[2,1] - (x[2,1]*x[2,2]*x[1,2]))

out = img.copy()
out[S == 0] = np.array((  0,   0, 255))
out[S == 1] = np.array((  0, 255,   0))
out[S == 2] = np.array((255,   0,   0))
out[S == 3] = np.array((255, 255,   0))
out[S == 4] = np.array((255,   0, 255))

cv2.imwrite("my_answer_61.png", out.astype(np.uint8))
