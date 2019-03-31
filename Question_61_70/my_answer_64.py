import cv2
import numpy as np

def connect_num(x):
    x = 1 - x
    return (x[1,2] - (x[1,2]*x[0,2]*x[0,1])) +\
           (x[0,1] - (x[0,1]*x[0,0]*x[1,0])) +\
           (x[1,0] - (x[1,0]*x[2,0]*x[2,1])) +\
           (x[2,1] - (x[2,1]*x[2,2]*x[1,2]))

def connect_array(img):
    H, W, C = img.shape
    tmp = np.zeros((H, W), dtype=np.uint8)
    tmp[np.sum(img, axis=2) != 0] = 1
    tmp2 = np.pad(tmp, (1, 1), 'edge')
    S = np.full((H, W), -1, dtype=np.float)
    for j in range(H):
        for i in range(W):
            if tmp[j, i] == 0:
                continue
            x = tmp2[j:j+3, i:i+3]
            S[j, i] = connect_num(x)

    return S

img = cv2.imread("gazo.png")
H, W, C = img.shape

while True:
    count = 0

    tmp = np.zeros((H, W), dtype=np.int)
    tmp[np.sum(img, axis=2) != 0] = 1
    tmp2 = np.zeros((H+2, W+2), dtype=np.int)
    tmp2[1:H+1, 1:W+1] = 1 - tmp
    K1 = np.array(((0, 1, 0),
                   (1, 0, 1),
                   (0, 1, 0)))
        
    S = connect_array(img)

    tmp3 = np.zeros((H+2, W+2), dtype=np.int)
    tmp3[1:H+1, 1:W+1] = tmp
    K3 = np.array(((1, 1, 1),
                   (1, 0, 1),
                   (1, 1, 1)))

    for y in range(H):
        for x in range(W):
            if tmp[y, x] == 0:
                continue
            # 1. 注目画素の4近傍に0が一つ以上存在する
            if np.sum(tmp2[y:y+3, x:x+3] * K1) < 1:
                continue
            # 2. x0の8近傍に1or-1が2つ以上存在する
            if np.sum(np.abs(tmp3[y:y+3, x:x+3]) * K3) < 2:
                continue
            # 3. x0の8近傍に1が1つ以上存在する
            if np.any((tmp3[y:y+3, x:x+3] * K3) == 1) == False:
                continue
            # 4. x0の8-連結数が1である
            if S[y, x] != 1:
                continue
            # 5. xn(n=1〜8)全てに対して以下のどちらかが成り立つ
            # - xnが-1以外
            # - xnを0とした時、x0の8-連結数が1である
            flag = False
            for j in range(-1, 1):
                for i in range(-1, 1):
                    if tmp3[y+1+j, x+1+i] != -1:
                        continue
                    tmp3[y+1+j, x+1+i] = 0
                    s = connect_num(tmp3[y:y+3, x:x+3])
                    tmp3[y+1+j, x+1+i] = -1
                    if s == 1:
                        continue
                    flag = True # どちらも成立しなかった
            if flag:
                continue
            tmp[y, x] = -1
            tmp3[y+1, x+1] = -1
            count += 1

    img[tmp == -1] = np.array((0, 0, 0))
    if count == 0:
        break

cv2.imwrite("my_answer_64.png", img.astype(np.uint8))
