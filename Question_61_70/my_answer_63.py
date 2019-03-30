import cv2
import numpy as np

def connect_num(img):
    H, W, C = img.shape
    tmp = np.zeros((H, W), dtype=np.uint8)
    tmp[np.sum(img, axis=2) != 0] = 1
    tmp2 = np.pad(tmp, (1, 1), 'edge')
    S = np.full((H, W), -1, dtype=np.float)
    for j in range(H):
        for i in range(W):
            if tmp[j, i] == 0:
                continue
            x = tmp2[j:j+3, i:i+3]
            S[j, i] = (x[1,2] - (x[1,2]*x[0,2]*x[0,1])) +\
                      (x[0,1] - (x[0,1]*x[0,0]*x[1,0])) +\
                      (x[1,0] - (x[1,0]*x[2,0]*x[2,1])) +\
                      (x[2,1] - (x[2,1]*x[2,2]*x[1,2]))

    return S

img = cv2.imread("gazo.png")
H, W, C = img.shape

while True:
    count = 0

    tmp = np.zeros((H, W), dtype=np.uint8)
    tmp[np.sum(img, axis=2) != 0] = 1
    tmp2 = np.zeros((H+2, W+2), dtype=np.uint8)
    tmp2[1:H+1, 1:W+1] = 1 - tmp
    K1 = np.array(((0, 1, 0),
                   (1, 0, 1),
                   (0, 1, 0)))
        
    S = connect_num(img)

    tmp3 = np.zeros((H+2, W+2), dtype=np.uint8)
    tmp3[1:H+1, 1:W+1] = tmp
    K3 = np.array(((1, 1, 1),
                   (1, 0, 1),
                   (1, 1, 1)))

    out = img.copy()
    for y in range(H):
        for x in range(W):
            if tmp[y, x] == 0:
                continue
            if np.sum(tmp2[y:y+3, x:x+3] * K1) < 1:
                continue
            if S[y, x] != 1:
                continue
            if np.sum(tmp3[y:y+3, x:x+3] * K3) < 3:
                continue
            out[y, x] = np.array((0, 0, 0))
            count += 1

    img = out.copy()
    if count == 0:
        break

cv2.imwrite("my_answer_63.png", out.astype(np.uint8))
