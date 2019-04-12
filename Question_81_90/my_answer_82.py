import cv2
import numpy as np
import matplotlib.pyplot as plt

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

M = np.zeros((H*2, W*2), dtype=np.float)
M[0:H, 0:W] = Ix * Ix
M[0:H, W:W*2] = Ix * Iy
M[H:H*2, 0:W] = M[0:H, W:W*2]
M[H:H*2, W:W*2] = Iy * Iy

# 3.Ix^2, Iy^2, IxIyにそれぞれガウシアンフィルターをかける。
tmpIx = np.zeros((H+2, W+2), dtype=np.float)
tmpIx[1:-1, 1:-1] = M[0:H, 0:W]
tmpIy = np.zeros((H+2, W+2), dtype=np.float)
tmpIy[1:-1, 1:-1] = M[H:H*2, W:W*2]
tmpIxy = np.zeros((H+2, W+2), dtype=np.float)
tmpIxy[1:-1, 1:-1] = M[0:H, W:W*2]

kernel = np.zeros((K_size, K_size), dtype=np.float)
for x in range(K_size):
    for y in range(K_size):
        kernel[y, x] = g(x-1, y-1, 3)
kernel /= kernel.sum()

for y in range(H):
    for x in range(W):
        M[y, x] = np.sum(tmpIx[y:y+K_size, x:x+K_size] * kernel)
        M[y+H, x+W] = np.sum(tmpIy[y:y+K_size, x:x+K_size] * kernel)
        M[y+H, x] = np.sum(tmpIxy[y:y+K_size, x:x+K_size] * kernel)
        M[y, x+W] = M[y+H, x]

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)
plt.gray()
fig, axes = plt.subplots(1, 3, subplot_kw={'xticks': [], 'yticks': []})
axes[0].imshow(M[0:H, 0:W])
axes[0].set_title("Ix^2")
axes[1].imshow(M[H:H*2, W:W*2])
axes[1].set_title("Iy^2")
axes[2].imshow(M[0:H, W:W*2])
axes[2].set_title("IxIy")
plt.savefig("my_answer_82.png")
