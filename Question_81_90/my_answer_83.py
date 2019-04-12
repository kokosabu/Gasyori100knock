import cv2
import numpy as np
import numpy.linalg as LA

K_size = 3
pad = K_size // 2
Kx = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])
Ky = np.array([[ 1,  2,  1],
               [ 0,  0,  0],
               [-1, -2, -1]])

def g(x, y, s):
    return 1 / (s*np.sqrt(2 * np.pi)) * np.exp( - (x**2 + y**2) / (2*s**2))

img = cv2.imread("thorino.jpg").astype(np.float)
H, W, C = img.shape

# 1.画像をグレースケール化。
gray = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]

# 2.Sobelフィルタにより、ヘシアン行列を求める。
tmp = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
tmp[pad:pad+H, pad:pad+W] = gray

Ix = gray.copy()
Iy = gray.copy()
for y in range(H):
    for x in range(W):
        Iy[y, x] = np.mean(Ky * tmp[y:y+K_size, x:x+K_size])
        Ix[y, x] = np.mean(Kx * tmp[y:y+K_size, x:x+K_size])

M = np.zeros((H, W, 2, 2), dtype=np.float)
M[..., 0, 0] = Ix * Ix
M[..., 0, 1] = Ix * Iy
M[..., 1, 0] = M[..., 0, 1]
M[..., 1, 1] = Iy * Iy

# 3.Ix^2, Iy^2, IxIyにそれぞれガウシアンフィルターをかける。
tmpIx  = np.pad(Ix*Ix, (1, 1), 'edge')
tmpIy  = np.pad(Iy*Iy, (1, 1), 'edge')
tmpIxy = np.pad(Ix*Iy, (1, 1), 'edge')

kernel = np.zeros((K_size, K_size), dtype=np.float)
for x in range(K_size):
    for y in range(K_size):
        kernel[y, x] = g(x-1, y-1, 3)
kernel /= kernel.sum()

for y in range(H):
    for x in range(W):
        M[y, x, 0, 0] = np.sum(tmpIx[y:y+K_size, x:x+K_size] * kernel)
        M[y, x, 1, 1] = np.sum(tmpIy[y:y+K_size, x:x+K_size] * kernel)
        M[y, x, 0, 1] = np.sum(tmpIxy[y:y+K_size, x:x+K_size] * kernel)
        M[y, x, 1, 0] = M[y, x, 0, 1]

# 4.各ピクセル毎に、R = det(H) - k (trace(H))^2 を計算する。
#   (kは実験的に0.04 - 0.16らへんが良いとされる)
R = LA.det(M) - 0.04 * np.trace(M, axis1=2, axis2=3)**2

# 5.R >= max(R) * th を満たすピクセルがコーナーとなる。 (thは0.1となることが多い)
out = img.copy()
out[...,2] = gray
out[...,1] = gray
out[...,0] = gray
out[R >= np.max(R)*0.1] = (0, 0, 255)

cv2.imwrite("my_answer_83.jpg", out.astype(np.uint8))
