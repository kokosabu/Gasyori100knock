import cv2
import numpy as np

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

a = 1.3
b = 0
c = 0
d = 0.8
tx = 0
ty = 0

H_new = int(np.ceil(H*d))
W_new = int(np.ceil(W*a))

print(H_new)
print(W_new)

y = np.arange(H).repeat(W).reshape(W, -1)
x = np.tile(np.arange(W), (H, 1))

out = np.zeros((H_new+1, W_new+1, C), dtype=np.float)

x_new = a * x + b * y + tx
y_new = c * x + d * y + ty

x_new = np.minimum(np.maximum(x_new, 0), W_new).astype(np.int)
y_new = np.minimum(np.maximum(y_new, 0), H_new).astype(np.int)

out[y_new, x_new] = img[y, x]
out = out[:H_new, :W_new]

cv2.imwrite("my_answer_29_1.jpg", out.astype(np.uint8))

tx = 30
ty = -30

out2 = np.zeros((H_new+1, W_new+1, C), dtype=np.float)

x_new = a * x + b * y + tx
y_new = c * x + d * y + ty

x_new = np.minimum(np.maximum(x_new, 0), W_new).astype(np.int)
y_new = np.minimum(np.maximum(y_new, 0), H_new).astype(np.int)

out2[y_new, x_new] = img[y, x]
out2 = out2[:H_new, :W_new]
cv2.imwrite("my_answer_29_2.jpg", out2.astype(np.uint8))
