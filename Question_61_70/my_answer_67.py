import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

# 1. 画像をグレースケール化し、x、ｙ方向の輝度勾配を求める
gray = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]
gray2 = np.pad(gray, (1, 1), 'edge')

# 2. gx, gyから勾配強度と勾配角度を求める。
gx = gray2[1:H+1, 2:W+2] - gray2[1:H+1, 0:W]
gx[gx == 0] = 0.000001
gy = gray2[2:H+2, 1:W+1] - gray2[0:H, 2:W+2]

mag = np.sqrt(gx ** 2 + gy ** 2)
ang = np.arctan(gy / gx) * (180/np.pi)
ang[ang < 0] += 360
ang[ang > 180] -= 180

# 3. 勾配角度を [0, 180]で9分割した値に量子化する。
#    つまり、[0,20]には0、[20, 40]には1というインデックスを求める。
for i in range(9):
    ang[np.where((ang >= i*20) & (ang < (i+1)*20), True, False)] = i
ang[ang == 180] = 8

# 4. 画像をN x Nの領域に分割し(この領域をセルという)、
#    セル内で3で求めたインデックスのヒストグラムを作成する。
#    ただし、当表示は1でなく勾配角度を求める。
N = 8
ang = ang.astype(np.int)

cell_H = H // N
cell_W = W // N

hist = np.zeros((9, cell_H, cell_W), dtype=np.float)
for y in range(H):
    for x in range(W):
        hist[ang[y, x], y//N, x//N] += mag[y, x]

fig, axes = plt.subplots(3, 3, subplot_kw={'xticks': [], 'yticks': []})
for ax, i in zip(axes.flat, range(9)):
    ax.imshow(hist[i])
plt.savefig("my_answer_67.png")

# 5. C x Cのセルを１つとして(これをブロックという)、
#    ブロック内のセルのヒストグラムを次式で正規化する。
#    これを1セルずつずらしながら行うので、一つのセルが何回も正規化される。
