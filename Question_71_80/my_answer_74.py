import cv2
import numpy as np

pad = 1

# グレースケール向け
def bi_linear(img, a):
    H, W = img.shape
    #H, W, C = img.shape
    Ha = int(H*a)
    Wa = int(W*a)

    in_ = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
    #in_ = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float)
    in_[pad:H+pad, pad:W+pad] = img

    out = np.zeros((Ha, Wa), dtype=np.float)
    #out = np.zeros((Ha, Wa, C), dtype=np.float)

    for y in range(Ha):
        for x in range(Wa):
            (xa, ya) = (x/a, y/a)
            
            dx = xa - int(xa)
            dy = ya - int(ya)

            out[y, x] = (1-dx)*(1-dy)*in_[int(y/a)+pad,   int(x/a)+pad  ] + \
                            dx*(1-dy)*in_[int(y/a)+pad,   int(x/a)+1+pad] + \
                        (1-dx)*    dy*in_[int(y/a)+1+pad, int(x/a)+pad  ] + \
                            dx*    dy*in_[int(y/a)+pad,   int(x/a)+pad  ]

    return out

img = cv2.imread("imori.jpg").astype(np.float)
gray = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]

out = bi_linear(gray, 0.5)
out = bi_linear(out, 2.0)
out = np.abs(gray - out)
out = np.round((out - np.min(out)) * (255 / (np.max(out) - np.min(out))))
out = out.astype(np.uint8)

cv2.imwrite("my_answer_74.jpg", out)
