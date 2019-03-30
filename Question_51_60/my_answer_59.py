import cv2
import numpy as np

img = cv2.imread("seg.png").astype(np.float)
H, W, C = img.shape

label = 0
pad = 1
table = {}
K = np.array(((1, 1, 1),
              (1, 0, 0),
              (0, 0, 0)))

labels = np.zeros((H+1, W+1), dtype=np.uint)
for y in range(H):
    for x in range(W):
        if np.all(img[y, x] == 0):
            continue
        l = labels[y:y+3, x:x+3] * K
        lsum = np.sum(l)
        l[l == 0] = label + 1
        lmin = np.min(l)
        if lsum == 0:
            label += 1
            table[label] = label
            labels[y+pad, x+pad] = label
        else:
            labels[y+pad, x+pad] = lmin
            for j in range(3):
                for i in range(3):
                    if l[j, i] != (label + 1):
                        table[l[j, i]] = table[lmin]

for k in table.keys():
    labels[labels == k] = table[k]

out = img.copy()
labels = labels[1:H+1, 1:W+1]
for l in range(1, label+1):
    out[labels == l] = (255, 0, l*30)

cv2.imwrite("my_answer_59.png", out.astype(np.uint8))
