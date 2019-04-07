import cv2
import numpy as np

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

# 5. C x Cのセルを１つとして(これをブロックという)、
#    ブロック内のセルのヒストグラムを次式で正規化する。
#    これを1セルずつずらしながら行うので、一つのセルが何回も正規化される。
C = 3
epsilon = 1

hist2 = np.zeros((9, cell_H, cell_W), dtype=np.float)
for y in range(cell_H - C + 1):
    for x in range(cell_W - C + 1):
        for i in range(9):
            hist2[i, y:y+C, x:x+C] += hist[i, y:y+C, x:x+C] / np.sqrt(np.sum(hist[i, y:y+C, x:x+C]) + epsilon)

hist = hist2
hist = (hist-np.min(hist))*(255/(np.max(hist)-np.min(hist)))
hist = hist.astype(np.uint8)


pat = np.zeros((9, N, N), dtype=np.float)
for i in range(9):
    m = np.tan((i * 20) * np.pi/180)
    m += 0.0001
    for y in range(-cell_H//2, cell_H//2):
        x = np.round(y / m)
        nx = (x + np.round(N/2)).astype(np.uint8)
        ny = (y + np.round(N/2)).astype(np.uint8)
        if nx >= 0 and nx < N and ny >= 0 and ny < N:
            pat[i, ny, nx] = 1
    for x in range(-cell_W//2, cell_W//2):
        y = np.round(x * m)
        nx = (x + np.round(N/2)).astype(np.uint8)
        ny = (y + np.round(N/2)).astype(np.uint8)
        if nx >= 0 and nx < N and ny >= 0 and ny < N:
            pat[i, ny, nx] = 1

out = np.zeros((H, W), dtype=np.float)
out2 = gray.copy()
for i in range(9):
    for y in range(cell_H):
        for x in range(cell_W):
            out[y*N:(y+1)*N, x*N:(x+1)*N] += pat[i] * hist[i, y, x]
            out2[y*N:(y+1)*N, x*N:(x+1)*N] += pat[i] * hist[i, y, x]

out[out > 255] = 255
out2[out2 > 255] = 255
cv2.imwrite("my_answer_69_hog.jpg", out.astype(np.uint8))
cv2.imwrite("my_answer_69.jpg", out2.astype(np.uint8))
