import cv2
import numpy as np

_img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = _img.shape
img = np.zeros((H+2, W+2, C), dtype=np.float)
img[1:H+1, 1:W+1] = _img

#---- 1 ----

dx = 30
dy = 0
a = 1
b = dx / H
c = 0
d = 1
tx = 0
ty = 0

A = np.array([[a, b, tx],
              [c, d, ty],
              [0, 0,  1]])
A_inv = np.linalg.inv(A)

W_new = W + dx
H_new = H + dy

y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)
x_new = np.tile(np.arange(W_new), (H_new, 1))

x = A_inv[0][0] * x_new + A_inv[0][1] * y_new + A_inv[0][2] + 1
y = A_inv[1][0] * x_new + A_inv[1][1] * y_new + A_inv[1][2] + 1
out1 = np.zeros((H_new+1, W_new+1, C), dtype=np.float)
x_new = np.minimum(np.maximum(x_new, 0), W_new).astype(np.int)
y_new = np.minimum(np.maximum(y_new, 0), H_new).astype(np.int)
x1 = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
y1 = np.minimum(np.maximum(y, 0), H+1).astype(np.int)
out1[y_new, x_new] = img[y1, x1]
out1 = out1[:H_new, :W_new]
cv2.imwrite("my_answer_31_1.jpg", out1.astype(np.uint8))

#---- 2 ----

dx = 0
dy = 30
a = 1
b = 0
c = dy / W
d = 1
tx = 0
ty = 0

A = np.array([[a, b, tx],
              [c, d, ty],
              [0, 0,  1]])
A_inv = np.linalg.inv(A)

W_new = W + dx
H_new = H + dy

y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)
x_new = np.tile(np.arange(W_new), (H_new, 1))

x = A_inv[0][0] * x_new + A_inv[0][1] * y_new + A_inv[0][2] + 1
y = A_inv[1][0] * x_new + A_inv[1][1] * y_new + A_inv[1][2] + 1
out2 = np.zeros((H_new+1, W_new+1, C), dtype=np.float)
x_new = np.minimum(np.maximum(x_new, 0), W_new).astype(np.int)
y_new = np.minimum(np.maximum(y_new, 0), H_new).astype(np.int)
x1 = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
y1 = np.minimum(np.maximum(y, 0), H+1).astype(np.int)
out2[y_new, x_new] = img[y1, x1]
out2 = out2[:H_new, :W_new]
cv2.imwrite("my_answer_31_2.jpg", out2.astype(np.uint8))

#---- 3 ----

dx = 30
dy = 30
a = 1
b = dx / H
c = dy / W
d = 1
tx = 0
ty = 0

A = np.array([[a, b, tx],
              [c, d, ty],
              [0, 0,  1]])
A_inv = np.linalg.inv(A)

W_new = W + dx
H_new = H + dy

y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)
x_new = np.tile(np.arange(W_new), (H_new, 1))

x = A_inv[0][0] * x_new + A_inv[0][1] * y_new + A_inv[0][2] + 1
y = A_inv[1][0] * x_new + A_inv[1][1] * y_new + A_inv[1][2] + 1
out3 = np.zeros((H_new+1, W_new+1, C), dtype=np.float)
x_new = np.minimum(np.maximum(x_new, 0), W_new).astype(np.int)
y_new = np.minimum(np.maximum(y_new, 0), H_new).astype(np.int)
x1 = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
y1 = np.minimum(np.maximum(y, 0), H+1).astype(np.int)
out3[y_new, x_new] = img[y1, x1]
out3 = out3[:H_new, :W_new]
cv2.imwrite("my_answer_31_3.jpg", out3.astype(np.uint8))
#y -= int(y[H//2][W//2] - H/2)
#x -= int(x[H//2][W//2] - W/2)
#x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
#y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)
#out[y_new, x_new] = img[y, x]
#out = out[:H_new, :W_new]
#cv2.imwrite("my_answer_30_2.jpg", out.astype(np.uint8))
