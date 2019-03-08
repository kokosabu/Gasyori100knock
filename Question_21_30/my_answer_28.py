import cv2
import numpy as np

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

a = 1
b = 0
c = 0
d = 1
tx = 30
ty = -30

y = np.arange(H).repeat(W).reshape(W, -1)
x = np.tile(np.arange(W), (H, 1))

out = np.zeros((H+1, W+1, C), dtype=np.float)

x_new = a * x + b * y + tx
y_new = c * x + d * y + ty

x_new = np.minimum(np.maximum(x_new, 0), W).astype(np.int)
y_new = np.minimum(np.maximum(y_new, 0), H).astype(np.int)

out[y_new, x_new] = img[y, x]
out = out[:H, :W]

cv2.imwrite("my_answer_28.jpg", out.astype(np.uint8))
