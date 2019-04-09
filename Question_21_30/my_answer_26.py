import cv2
import numpy as np

pad = 1

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

a = 1.5
Ha = int(H*a)
Wa = int(W*a)

in_ = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float)
in_[pad:H+pad, pad:W+pad] = img

out = np.zeros((Ha, Wa, C), dtype=np.float)

for y in range(Ha):
    for x in range(Wa):
        #(xa, ya) = np.floor((x/a, y/a))
        (xa, ya) = (x/a, y/a)
        
        dx = xa - int(xa)
        dy = ya - int(ya)

        out[y, x] = (1-dx)*(1-dy)*in_[int(y/a)+pad,   int(x/a)+pad  ] + \
                        dx*(1-dy)*in_[int(y/a)+pad,   int(x/a)+1+pad] + \
                    (1-dx)*    dy*in_[int(y/a)+1+pad, int(x/a)+pad  ] + \
                        dx*    dy*in_[int(y/a)+1+pad, int(x/a)+1+pad]

out[out < 0] = 0
out[out > 255] = 255
out = out.astype(np.uint8)

cv2.imwrite("my_answer_26.jpg", out)
