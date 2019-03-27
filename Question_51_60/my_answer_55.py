import cv2
import numpy as np

img = cv2.imread("imori.jpg").astype(np.float)
part = cv2.imread("imori_part.jpg").astype(np.float)

H, W, _ = img.shape
h, w, _ = part.shape

S = np.zeros((H-h, W-w), dtype=np.float)
for j in range(H-h):
    for i in range(W-w):
        S[j, i] = np.sum(np.abs((img[j:j+h, i:i+w] - part)))
y, x = np.where(S == np.min(S))

out = img.copy()
cv2.rectangle(out, (x[0], y[0]), (x[0]+w, y[0]+h), (0, 0, 255), thickness=1)
#out[y[0]:y[0]+h, x[0]]        = (0, 0, 255)
#out[y[0]:y[0]+h, x[0]+w]      = (0, 0, 255)
#out[y[0],        x[0]:x[0]+w] = (0, 0, 255)
#out[y[0]+h,      x[0]:x[0]+w] = (0, 0, 255)

cv2.imwrite("my_answer_55.jpg", out.astype(np.uint8))
