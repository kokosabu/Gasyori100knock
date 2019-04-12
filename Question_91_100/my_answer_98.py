import cv2
import numpy as np

pad = 1

class NN:
    def __init__(self, ind=144, w=64, w2=64, outd=1, lr=0.01):
        self.w1 = np.random.normal(0, 1, [ind, w])
        self.b1 = np.random.normal(0, 1, [w])
        self.w2 = np.random.normal(0, 1, [w, w2])
        self.b2 = np.random.normal(0, 1, [w2])
        self.wout = np.random.normal(0, 1, [w2, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr

    def forward(self, x):
        self.z1 = x
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        grad_En = (self.out - t) * self.out * (1 - self.out)
        grad_wout = np.dot(self.z3.T, grad_En)
        grad_bout = np.dot(np.ones([grad_En.shape[0]]), grad_En)
        self.wout -= self.lr * grad_wout
        self.bout -= self.lr * grad_bout

        # backpropagation inter layer
        grad_u2 = np.dot(grad_En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2

        #grad_u1 = np.dot(En, self.z3.T) * self.z2 * (1 - self.z2)
        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

def sigmoid(x):
    sigmoid_range = 34.538776394910683
    x[x<=-sigmoid_range] = sigmoid_range
    x[x>=sigmoid_range] = -sigmoid_range
    return 1. / (1. + np.exp(-x))

def iou(a, b):
    start_x = np.max((a[0], b[0]))
    start_y = np.max((a[1], b[1]))
    end_x = np.min((a[2], b[2]))
    end_y = np.min((a[3], b[3]))
    w = end_x - start_x
    h = end_y - start_y
    if w < 0:
        w = 0
    if h < 0:
        h = 0

    R1 = (a[2]-a[0]) * (a[3]-a[1])
    R2 = (b[2]-b[0]) * (b[3]-b[1])
    Rol = w * h

    return Rol / (R1 + R2 - Rol)

def bi_linear(img, ha, wa):
    H, W = img.shape
    #H, W, C = img.shape
    Ha = int(H*ha)
    Wa = int(W*wa)

    in_ = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
    #in_ = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float)
    in_[pad:H+pad, pad:W+pad] = img

    out = np.zeros((Ha, Wa), dtype=np.float)
    #out = np.zeros((Ha, Wa, C), dtype=np.float)

    for y in range(Ha):
        for x in range(Wa):
            (xa, ya) = (x/wa, y/ha)
            
            dx = xa - int(xa)
            dy = ya - int(ya)

            out[y, x] = (1-dx)*(1-dy)*in_[int(y/ha)+pad,   int(x/wa)+pad  ] + \
                            dx*(1-dy)*in_[int(y/ha)+pad,   int(x/wa)+1+pad] + \
                        (1-dx)*    dy*in_[int(y/ha)+1+pad, int(x/wa)+pad  ] + \
                            dx*    dy*in_[int(y/ha)+pad,   int(x/wa)+pad  ]

    return out

def hog(resize):
    rS = 32
    # 1. 画像をグレースケール化し、x、ｙ方向の輝度勾配を求める
    gray2 = np.pad(resize, (1, 1), 'edge')
    gx = gray2[1:rS+1, 2:rS+2] - gray2[1:rS+1, 0:rS]
    gx[gx == 0] = 0.000001
    gy = gray2[2:rS+2, 1:rS+1] - gray2[0:rS, 2:rS+2]
    # 2. gx, gyから勾配強度と勾配角度を求める。
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan(gy / gx) * (180/np.pi)
    ang[ang < 0] += 360
    ang[ang > 180] -= 180
    # 3. 勾配角度を [0, 180]で9分割した値に量子化する。
    for i in range(9):
        ang[np.where((ang >= i*20) & (ang < (i+1)*20), True, False)] = i
    ang[ang == 180] = 8
    # 4. 画像をN x Nの領域に分割し(この領域をセルという)、
    #    セル内で3で求めたインデックスのヒストグラムを作成する。
    N = 8
    ang = ang.astype(np.int)
    cell_H = rS // N
    cell_W = rS // N
    hist = np.zeros((9, cell_H, cell_W), dtype=np.float)
    for y in range(rS):
        for x in range(rS):
            hist[ang[y, x], y//N, x//N] += mag[y, x]
    # 5. C x Cのセルを１つとして(これをブロックという)、
    #    ブロック内のセルのヒストグラムを次式で正規化する。
    C = 3
    epsilon = 1
    hist2 = np.zeros((9, cell_H, cell_W), dtype=np.float)
    for y in range(cell_H - C + 1):
        for x in range(cell_W - C + 1):
            for i in range(9):
                hist2[i, y:y+C, x:x+C] += hist[i, y:y+C, x:x+C] / np.sqrt(np.sum(hist[i, y:y+C, x:x+C]) + epsilon)
    hist = (hist2-np.min(hist2))*(255/(np.max(hist2)-np.min(hist2)))
    return hist.reshape((144))

### main
rS = 32
np.random.seed(0)

img = cv2.imread("imori_1.jpg").astype(np.float)
H, W, C = img.shape
gray = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]

gt = np.array((47, 41, 129, 103), dtype=np.float32)

features = np.zeros((200, 144), dtype=np.float)
labels = np.zeros((200, 1), dtype=np.int)
for i in range(200):
    crop = np.zeros((4), dtype=np.int)
    crop[0] = np.random.randint(W-60)
    crop[1] = np.random.randint(H-60)
    crop[2] = crop[0] + 60
    crop[3] = crop[1] + 60
    resize = bi_linear(gray[crop[1]:crop[3], crop[0]:crop[2]], rS/60, rS/60)
    features[i] = hog(resize)
    if iou(crop, gt) >= 0.5:
        labels[i][0] = 1

train_x = features
train_t = labels

nn = NN()

# train
for i in range(10000):
    nn.forward(train_x)
    nn.train(train_x, train_t)

img = cv2.imread("imori_many.jpg").astype(np.float)
H, W, C = img.shape
gray = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]

recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.int)
detects = np.ndarray((0, 5), dtype=np.float32)
for y in range(0, H, 4):
    for x in range(0, W, 4):
        for rec in recs:
            dh = int(rec[0] // 2)
            dw = int(rec[1] // 2)
            x1 = max(x-dw, 0)
            x2 = min(x+dw, W)
            y1 = max(y-dh, 0)
            y2 = min(y+dh, H)
            crop = gray[y1:y2, x1:x2]
            ch, cw = crop.shape
            resize = bi_linear(crop, rS/ch, rS/cw)
            feature = hog(resize)
            p = nn.forward(feature)
            if p >= 0.7:
                detects = np.vstack((detects, np.array((x1, y1, x2, y2, p[0]))))

print(detects)
d = detects
out = img.copy()
for i in range(len(d)):
    out = cv2.rectangle(out, (int(d[i][0]), int(d[i][1])), (int(d[i][2]), int(d[i][3])), (0, 0, 255), 1)

cv2.imwrite("my_answer_98.jpg", out.astype(np.uint8))
