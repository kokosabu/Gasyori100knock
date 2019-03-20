import cv2
import numpy as np

def g(x, y, s):
    return 1 / (s*np.sqrt(2 * np.pi)) * np.exp( - (x**2 + y**2) / (2*s**2))

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape


# 1 ガウシアンフィルタを掛ける

# 1-1 画像をグレースケール化する
gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

# 1-2 ガウシアンフィルタ(5x5, s=1.4)をかける
tmp = np.pad(gray, (2, 2), 'edge')
out = np.pad(gray, (2, 2), 'edge')

kernel = np.zeros((5, 5), dtype=np.float)
for x in range(5):
    for y in range(5):
        kernel[y, x] = g(x-2, y-2, 1.4)
kernel /= kernel.sum()

for i in range(H):
    for j in range(W):
        out[i+2, j+2] = np.sum(tmp[i:i+5, j:j+5] * kernel)


# 2 x, y方向のSobelフィルタを掛け、それらからエッジ強度とエッジ勾配を求める
# 2-1 x方向、y方向のsobelフィルタを掛け、画像の勾配画像fx, fyを求め、勾配強度と勾配角度を次式で求める。
out = out[1:H+3, 1:W+3]
K_size = 3
Kh = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])
Kv = np.array([[ 1,  2,  1],
               [ 0,  0,  0],
               [-1, -2, -1]])

fy = gray.copy()
fx = gray.copy()
for y in range(H):
    for x in range(W):
        fy[y, x] = np.sum(Kv * out[y:y+K_size, x:x+K_size])
        fx[y, x] = np.sum(Kh * out[y:y+K_size, x:x+K_size])

edge = np.sqrt(fx**2 + fy**2)
tan = np.arctan(fy / fx)


# 2-2 勾配角度を次式に沿って、量子化する。
angle = np.zeros((H, W), dtype=np.float)
angle[(-0.4142 < tan) & (tan <= 0.4142)]  =   0
angle[(0.4142 < tan) & (tan < 2.4142)]  =  45
angle[np.abs(tan) >= 2.4142]           =  90
angle[(-2.4142 < tan) & (tan <= -0.4142)] = 135

# 3 エッジ勾配の値から、Non-maximum suppression によりエッジの細線化を行う

pad = 1
edge = np.pad(edge, (pad, pad), 'edge')
angle = np.pad(angle, (pad, pad), 'edge')
for y in range(H):
    for x in range(W):
        if angle[y+pad, x+pad] == 0:
            if edge[y+pad, x+pad] < edge[y+pad, x+pad-1] or edge[y+pad, x+pad] < edge[y+pad, x+pad+1]:
                edge[y+pad, x+pad] = 0
        if angle[y+pad, x+pad] == 45:
            if edge[y+pad, x+pad] < edge[y+pad+1, x+pad-1] or edge[y+pad, x+pad] < edge[y+pad-1, x+pad+1]:
                edge[y+pad, x+pad] = 0
        if angle[y+pad, x+pad] == 90:
            if edge[y+pad, x+pad] < edge[y+pad+1, x+pad] or edge[y+pad, x+pad] < edge[y+pad-1, x+pad]:
                edge[y+pad, x+pad] = 0
        if angle[y+pad, x+pad] == 135:
            if edge[y+pad, x+pad] < edge[y+pad+1, x+pad+1] or edge[y+pad, x+pad] < edge[y+pad-1, x+pad-1]:
                edge[y+pad, x+pad] = 0
edge = edge[pad:H+pad, pad:W+pad]


# 4 ヒステリシスによる閾値処理を行う
HT = 100
LT = 30

out = np.zeros((H, W), dtype=np.uint8)
#out = edge.copy()

# 4-1 勾配強度edge(x,y)がHTより大きい場合はedge(x,y)=255
out[edge > HT] = 255
#edge[edge > HT] = 255

# 4-2 LTより小さい場合はedge(x,y)=0
out[edge < LT] = 0
#edge[edge < LT] = 255

# 4-3 LT < edge(x,y) < HTの時、周り８ピクセルの勾配強度でHTより大きい値が存在すれば、edge(x,y)=255
edge = np.pad(edge, (1, 1), 'edge')
round8 = np.array(((-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)))
for y in range(H):
    for x in range(W):
        if LT > edge[y, x] or edge[y, x] > HT:
            continue
        out[y, x] = 0
        for r in range(8):
            if edge[y+pad+round8[r][0], x+pad+round8[r][1]] > HT:
                out[y, x] = 255
#edge = edge[pad:H+pad, pad:W+pad]

cv2.imwrite("my_answer_43.jpg", out.astype(np.uint8))
