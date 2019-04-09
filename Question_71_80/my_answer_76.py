import cv2
import numpy as np

pad = 1

# グレースケール向け
def bi_linear(img, a):
    H, W = img.shape
    Ha = int(H*a)
    Wa = int(W*a)

    in_ = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
    in_[pad:H+pad, pad:W+pad] = img

    out = np.zeros((Ha, Wa), dtype=np.float)

    for y in range(Ha):
        for x in range(Wa):
            (xa, ya) = (x/a, y/a)
            
            dx = xa - int(xa)
            dy = ya - int(ya)

            out[y, x] = (1-dx)*(1-dy)*in_[int(y/a)+  pad, int(x/a)+  pad] + \
                            dx*(1-dy)*in_[int(y/a)+  pad, int(x/a)+1+pad] + \
                        (1-dx)*    dy*in_[int(y/a)+1+pad, int(x/a)+  pad] + \
                            dx*    dy*in_[int(y/a)+1+pad, int(x/a)+1+pad]
    return out

img = cv2.imread("imori.jpg").astype(np.float)
gray = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]

pyramid = np.zeros((6, 128, 128))
for i in range(6):
    out = bi_linear(gray, 1 / 2**i)
    pyramid[i] = bi_linear(out, 2**i)

choice = np.array(((0,1), (0,3), (0,5), (1,4), (2,3), (3,5)))
diff_sum = np.zeros((128, 128), dtype=np.float)
for c in choice:
    diff_sum = diff_sum + np.abs(pyramid[c[0]] - pyramid[c[1]])

out = (diff_sum - np.min(diff_sum)) * (255 / (np.max(diff_sum) - np.min(diff_sum)))
cv2.imwrite("my_answer_76.jpg", np.round(out).astype(np.uint8))
