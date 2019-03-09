import cv2
import numpy as np

img_ = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img_.shape

img = np.zeros((H+2, W+2, C), dtype=np.float)
img[1:H+1, 1:W+1] = img_

a = 1.3
b = 0
c = 0
d = 0.8
tx = 0
ty = 0

A = np.array([[a, b, tx],
              [c, d, ty],
              [0, 0,  1]])
A_inv = np.linalg.inv(A)

H_new = int(np.ceil(H*d))
W_new = int(np.ceil(W*a))

x_new = np.tile(np.arange(W_new), (H_new, 1))
y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

out = np.zeros((H_new+1, W_new+1, C), dtype=np.float)

x = A_inv[0][0] * x_new + A_inv[0][1] * y_new + A_inv[0][2] + 1
y = A_inv[1][0] * x_new + A_inv[1][1] * y_new + A_inv[1][2] + 1

x_new = np.minimum(np.maximum(x_new, 0), W_new).astype(np.int)
y_new = np.minimum(np.maximum(y_new, 0), H_new).astype(np.int)
x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

out[y_new, x_new] = img[y, x]
out = out[:H_new, :W_new]
cv2.imwrite("my_answer_29_1.jpg", out.astype(np.uint8))

#----------

tx = 30
ty = -30

A = np.array([[a, b, tx],
              [c, d, ty],
              [0, 0,  1]])
A_inv = np.linalg.inv(A)

H_new = int(np.ceil(H*d))
W_new = int(np.ceil(W*a))

x_new = np.tile(np.arange(W_new), (H_new, 1))
y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

out2 = np.zeros((H_new+1, W_new+1, C), dtype=np.float)

x = A_inv[0][0] * x_new + A_inv[0][1] * y_new + A_inv[0][2] + 1
y = A_inv[1][0] * x_new + A_inv[1][1] * y_new + A_inv[1][2] + 1

x_new = np.minimum(np.maximum(x_new, 0), W_new).astype(np.int)
y_new = np.minimum(np.maximum(y_new, 0), H_new).astype(np.int)
x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

out2[y_new, x_new] = img[y, x]
out2 = out2[:H_new, :W_new]
cv2.imwrite("my_answer_29_2.jpg", out2.astype(np.uint8))
