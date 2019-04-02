import cv2
import numpy as np

img = cv2.imread("gazo.png")
H, W, C = img.shape

img2 = np.zeros((H, W), dtype=np.float)
img2[np.sum(img, axis=2) == 765] = 1

img2 = 1 - img2
tmp = np.ones((H+2, W+2), dtype=np.float)
tmp[1:H+1, 1:W+1] = img2

vect = np.array(((0, 1),
                 (0, 2),
                 (1, 2),
                 (2, 2),
                 (2, 1),
                 (2, 0),
                 (1, 0),
                 (0, 0),
                 (0, 1)))

K3 = np.array(((1, 1, 1),
               (1, 0, 1),
               (1, 1, 1)))
K4 = np.array(((0, 1, 0),
               (0, 0, 1),
               (0, 1, 0)))
K5 = np.array(((0, 0, 0),
               (1, 0, 1),
               (0, 1, 0)))
K4_2 = np.array(((0, 1, 0),
                 (1, 0, 1),
                 (0, 0, 0)))
K5_2 = np.array(((0, 1, 0),
                 (1, 0, 0),
                 (0, 1, 0)))

while True:
    count = 0

    tmp = np.ones((H+2, W+2), dtype=np.float)
    tmp[1:H+1, 1:W+1] = img2

    for y in range(H):
        for x in range(W):
            if tmp[y+1, x+1] == 1:
                continue
            change = 0
            old = -1
            for v in vect:
                now = tmp[y+v[0], x+v[1]]
                if old==0 and now==1:
                    change += 1
                old = now
            if change != 1:
                continue
            one_num = np.sum(tmp[y:y+3, x:x+3] * K3)
            if one_num < 2 or one_num > 6:
                continue
            if np.sum(tmp[y:y+3, x:x+3] * K4) < 1:
                continue
            if np.sum(tmp[y:y+3, x:x+3] * K5) < 1:
                continue

            img2[y, x] = 1
            count += 1

    tmp = np.ones((H+2, W+2), dtype=np.float)
    tmp[1:H+1, 1:W+1] = img2

    for y in range(H):
        for x in range(W):
            if tmp[y+1, x+1] == 1:
                continue
            change = 0
            old = -1
            for v in vect:
                now = tmp[y+v[0], x+v[1]]
                if old==0 and now==1:
                    change += 1
                old = now
            if change != 1:
                continue
            one_num = np.sum(tmp[y:y+3, x:x+3] * K3)
            if one_num < 2 or one_num > 6:
                continue
            if np.sum(tmp[y:y+3, x:x+3] * K4_2) < 1:
                continue
            if np.sum(tmp[y:y+3, x:x+3] * K5_2) < 1:
                continue

            img2[y, x] = 1
            count += 1

    if count == 0:
        break

img2 = 1 - img2
img2 = img2 * 255

cv2.imwrite("my_answer_65.png", img2.astype(np.uint8))
