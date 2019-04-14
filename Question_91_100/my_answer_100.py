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

def hog2(gray):
    h, w = gray.shape
    # Magnitude and gradient
    gray = np.pad(gray, (1, 1), 'edge')

    gx = gray[1:h+1, 2:] - gray[1:h+1, :w]
    gy = gray[2:, 1:w+1] - gray[:h, 1:w+1]
    gx[gx == 0] = 0.000001

    mag = np.sqrt(gx ** 2 + gy ** 2)
    gra = np.arctan(gy / gx)
    gra[gra<0] = np.pi / 2 + gra[gra < 0] + np.pi / 2

    # Gradient histogram
    gra_n = np.zeros_like(gra, dtype=np.int)

    d = np.pi / 9
    for i in range(9):
        gra_n[np.where((gra >= d * i) & (gra <= d * (i+1)))] = i

    N = 8
    HH = h // N
    HW = w // N
    Hist = np.zeros((HH, HW, 9), dtype=np.float32)
    for y in range(HH):
        for x in range(HW):
            for j in range(N):
                for i in range(N):
                    Hist[y, x, gra_n[y*4+j, x*4+i]] += mag[y*4+j, x*4+i]
                
    ## Normalization
    C = 3
    eps = 1
    for y in range(HH):
        for x in range(HW):
            #for i in range(9):
            Hist[y, x] /= np.sqrt(np.sum(Hist[max(y-1,0):min(y+2, HH), max(x-1,0):min(x+2, HW)] ** 2) + eps)

    #return Hist
    return Hist.reshape((144))

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
    features[i] = hog2(resize)
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
            feature = hog2(resize)
            p = nn.forward(feature)
            if p >= 0.7:
                detects = np.vstack((detects, np.array((x1, y1, x2, y2, p[0]))))

R = np.ndarray((0, 5), dtype=np.float32)
# 1.Boundinb-boxの集合Bをスコアが高い順にソートする。
B = detects[np.argsort(detects[:, 4])[::-1]]

#print("B", B)
t = 0.25
while True:
    # 2.スコアが最大のものをb0とする。
    b0 = B[0]
    # 3.b0と他のBounding-boxのIoUを計算する。IoUが閾値t以上のBounding-boxをBから削除する。B0は出力する集合Rに加え、Bから削除する。
    R = np.append(R, np.array([b0]), axis=0)
    del_list = []
    for i in range(len(B)):
        if iou(B[i, 0:4], b0[0:4]) >= t:
            del_list.append(i)
    B = np.delete(B, del_list, 0)
    if len(B) == 0:
        break
    # 4.2-3をBがなくなるまで行う。

# 5.Rを出力する。
#print("R", R)
out = img.copy()
#for i in range(len(R)):
#    out = cv2.rectangle(out, (int(R[i][0]), int(R[i][1])), (int(R[i][2]), int(R[i][3])), (0, 0, 255), 1)
#    out = cv2.putText(out, str(round(R[i][4], 2)), (int(R[i][0]), int(R[i][1])+9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

# [x1, y1, x2, y2]
t = 0.5
GT = np.array(((27, 48, 95, 110), (101, 75, 171, 138)), dtype=np.float32)
for gt in GT:
    out = cv2.rectangle(out, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 255, 0), 1)
cv2.imwrite("my_answer_100_gt.jpg", out.astype(np.uint8))

g_num = 0
for gt in GT:
    for i in range(len(R)):
        if iou(gt, R[i]) >= t:
            g_num += 1
            break
recall = g_num / len(GT)
print("Recall >> ", recall, "("+str(g_num)+" / "+str(len(GT))+")")

d_num = 0
for i in range(len(R)):
    for gt in GT:
        if iou(gt, R[i]) >= t:
            d_num += 1
            break
precision = d_num / len(R)
print("Precision >> ", precision, "("+str(d_num)+" / "+str(len(R))+")")

f_score = 2 * recall * precision / (recall + precision)
print("F-score >> ", f_score)

mAP = 0
j = 0
for i in range(len(R)):
    m = False
    for gt in GT:
        if iou(gt, R[i]) >= t:
            j += 1
            m = True
            mAP += j / (i+1)
            break
    if m == True:
        out = cv2.rectangle(out, (int(R[i][0]), int(R[i][1])), (int(R[i][2]), int(R[i][3])), (0, 0, 255), 1)
    else:
        out = cv2.rectangle(out, (int(R[i][0]), int(R[i][1])), (int(R[i][2]), int(R[i][3])), (255, 0, 0), 1)
    out = cv2.putText(out, str(round(R[i][4], 2)), (int(R[i][0]), int(R[i][1])+9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
mAP /= j
print("mAP >>", mAP)

cv2.imwrite("my_answer_100.jpg", out.astype(np.uint8))
