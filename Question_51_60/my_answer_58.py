import cv2
import numpy as np

img = cv2.imread("seg.png").astype(np.float)
H, W, C = img.shape

label = 0
pad = 1
table = {}

labels = np.zeros((H+1, W+1), dtype=np.uint8)
for y in range(H):
    for x in range(W):
        if np.all(img[y, x] == 0):
            continue
        if labels[y+pad-1, x+pad] == 0 and labels[y+pad, x+pad-1] == 0:
            label += 1
            table[label] = label
            labels[y+pad, x+pad] = label
        elif labels[y+pad-1, x+pad] == 0:
            labels[y+pad, x+pad] = labels[y+pad, x+pad-1]
        elif labels[y+pad, x+pad-1] == 0:
            labels[y+pad, x+pad] = labels[y+pad-1, x+pad]
        elif labels[y+pad-1, x+pad] < labels[y+pad, x+pad-1]:
            labels[y+pad, x+pad] = labels[y+pad-1, x+pad]
            table[labels[y+pad, x+pad-1]] = table[labels[y+pad-1, x+pad]]
        else:
            labels[y+pad, x+pad] = labels[y+pad, x+pad-1]
            table[labels[y+pad-1, x+pad]] = table[labels[y+pad, x+pad-1]]

for k in table.keys():
    labels[labels == k] = table[k]

out = img.copy()
labels = labels[1:H+1, 1:W+1]
for l in range(1, label+1):
    out[labels == l] = (255, 0, l*30)

cv2.imwrite("my_answer_58.png", out.astype(np.uint8))
