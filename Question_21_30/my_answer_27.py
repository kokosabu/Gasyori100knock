import cv2
import numpy as np

def h(t, a=-1):
    if np.abs(t) <= 1:
        return 1 - (a + 3)*np.abs(t)**2 + (a + 2)*np.abs(t)**3
    elif np.abs(t) <= 2:
        return -4*a + 8*a*np.abs(t) - 5*a*np.abs(t)**2 + a*np.abs(t)**3
    else:
        return 0

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

a = 1.5
Ha = int(H*a)
Wa = int(W*a)

in_ = np.zeros((H+3, W+3, C), dtype=np.float)
in_[1:H+1, 1:W+1] = img

out = np.zeros((Ha, Wa, C), dtype=np.float)

for ya in range(Ha):
    for xa in range(Wa):
        (y, x) = (ya/a, xa/a)
        indexes = np.array([(int(y)-1, int(x)-1),
                            (int(y)-1, int(x)  ),
                            (int(y)-1, int(x)+1),
                            (int(y)-1, int(x)+2),
                            (int(y)  , int(x)-1),
                            (int(y)  , int(x)  ),
                            (int(y)  , int(x)+1),
                            (int(y)  , int(x)+2),
                            (int(y)+1, int(x)-1),
                            (int(y)+1, int(x)  ),
                            (int(y)+1, int(x)+1),
                            (int(y)+1, int(x)+2),
                            (int(y)+2, int(x)-1),
                            (int(y)+2, int(x)  ),
                            (int(y)+2, int(x)+1),
                            (int(y)+2, int(x)+2)])
        sum_ = 0
        for t in range(16):
            dx = xa/a - indexes[t][1]
            dy = ya/a - indexes[t][0]
            out[ya, xa] += in_[indexes[t][0], indexes[t][1]] * h(dx) * h(dy)
            sum_ += h(dx) * h(dy)
        out[ya, xa] /= sum_

out[out < 0] = 0
out[out > 255] = 255
out = out.astype(np.uint8)

cv2.imwrite("my_answer_27.jpg", out)
