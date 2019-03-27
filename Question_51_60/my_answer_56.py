import cv2
import numpy as np

img = cv2.imread("imori.jpg").astype(np.float)
part = cv2.imread("imori_part.jpg").astype(np.float)

H, W, _ = img.shape
h, w, _ = part.shape

S = np.zeros((H-h, W-w), dtype=np.float)
for j in range(H-h):
    for i in range(W-w):
        S[j, i] = np.sum(img[j:j+h, i:i+w] * part) / (np.sqrt(np.sum(img[j:j+h, i:i+w]**2)) *  np.sqrt(np.sum(part**2)))
y, x = np.where(S == np.max(S))

out = img.copy()
cv2.rectangle(out, (x[0], y[0]), (x[0]+w, y[0]+h), (0, 0, 255), thickness=1)

cv2.imwrite("my_answer_56.jpg", out.astype(np.uint8))
