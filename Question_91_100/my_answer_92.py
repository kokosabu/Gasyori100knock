import cv2
import numpy as np

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

k = 5

flat_img = img.reshape((H*W, 3))

np.random.seed(0)
cls = flat_img[np.random.choice(np.arange(W*H), k, replace=False)]

indexs = np.zeros((H*W), dtype=np.int)
while True:
    for i in range(len(flat_img)):
        dis = np.zeros((len(cls)), dtype=np.int)
        for j in range(len(cls)):
            dis[j] = np.sqrt((flat_img[i, 2] - cls[j, 2])**2 + \
                             (flat_img[i, 1] - cls[j, 1])**2 + \
                             (flat_img[i, 0] - cls[j, 0])**2)
        indexs[i] = np.where(dis == np.min(dis))[0][0]

    newcls = cls.copy()
    for i in range(len(newcls)):
        newcls[i, 2] = np.mean(flat_img[indexs == i, 2])
        newcls[i, 1] = np.mean(flat_img[indexs == i, 1])
        newcls[i, 0] = np.mean(flat_img[indexs == i, 0])

    if (cls == newcls).all():
        break
    cls = newcls.copy()

out = img.copy()
indexs = indexs.reshape((H, W))
for i in range(len(cls)):
    out[indexs == i] = cls[i]
cv2.imwrite("my_answer_92.jpg", out.astype(np.uint8))
