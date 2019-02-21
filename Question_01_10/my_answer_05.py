import cv2
import numpy as np

img = cv2.imread("imori.jpg")

img = img.astype(np.float32)
r = img[:, :, 2] / 255
g = img[:, :, 1] / 255
b = img[:, :, 0] / 255

Max = r.copy()
Min = r.copy()
for i in range(len(Max)):
    Max[i] = np.maximum(r[i], g[i], b[i])
    Min[i] = np.minimum(r[i], g[i], b[i])
#Max = np.maximum(r, g, b)
#Min = np.minimum(r, g, b)

h = r.copy()
for i in range(len(h)):
    for j in range(len(h[i])):
        if Max[i][j] == Min[i][j]:
            h[i][j] = 0
        elif Min[i][j] == b[i][j]:
            h[i][j] = 60 * (g[i][j] - r[i][j]) / (Max[i][j] - Min[i][j]) + 60
        elif Min[i][j] == r[i][j]:
            h[i][j] = 60 * (b[i][j] - g[i][j]) / (Max[i][j] - Min[i][j]) + 180
        else:
            h[i][j] = 60 * (r[i][j] - b[i][j]) / (Max[i][j] - Min[i][j]) + 300

v = Max.copy()
s = Max.copy() - Min.copy()

#h = (h + 180) % 360
h += 180

c = s.copy()
#c = v * s
h_ = h / 60
x = c * (1 - np.fabs(h_ % 2 - 1))

for i in range(len(img)):
    for j in range(len(img[i])):
        tmp = (v[i][j] - c[i][j]) * np.array([1, 1, 1])
        print(v[i][j])
        print(c[i][j])
        print(v[i][j]-c[i][j])
        print(tmp)
        if h[i][j] == None:
            pass
        if 0 <= h_[i][j] and h_[i][j] < 1:
            tmp += np.array([c[i][j], x[i][j], 0])
        elif h_[i][j] < 2:
            tmp += np.array([x[i][j], c[i][j], 0])
        elif h_[i][j] < 3:
            tmp += np.array([0, c[i][j], x[i][j]])
        elif h_[i][j] < 4:
            tmp += np.array([0, x[i][j], c[i][j]])
        elif h_[i][j] < 5:
            tmp += np.array([x[i][j], 0, c[i][j]])
        else:
            tmp += np.array([c[i][j], 0, x[i][j]])
        #print(tmp)

        img[i,j,2] = tmp[0]
        img[i,j,1] = tmp[1]
        img[i,j,0] = tmp[2]

img = img * 255
img = img.astype(np.uint8)
cv2.imwrite("my_answer_05.jpg", img)

