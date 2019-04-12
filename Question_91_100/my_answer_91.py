import cv2
import numpy as np

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

k = 5

flat_img = img.reshape((H*W, 3))

np.random.seed(0)
cls = np.random.choice(np.arange(W*H), 5, replace=False)

indexs = np.zeros((H*W), dtype=np.int)
for i in range(len(flat_img)):
    dis = np.zeros((len(cls)), dtype=np.int)
    for j in range(len(cls)):
        dis[j] = np.sqrt((flat_img[i, 2] - flat_img[cls[j], 2])**2 + \
                         (flat_img[i, 1] - flat_img[cls[j], 1])**2 + \
                         (flat_img[i, 0] - flat_img[cls[j], 0])**2)
    indexs[i] = np.where(dis == np.min(dis))[0][0]

out = indexs.reshape((H, W)) * 50
cv2.imwrite("my_answer_91.jpg", out)


#for i in range(len(file_list)):
#    img = cv2.imread("dataset/" + file_list[i])
#    database[i, 0:12] = calc_hist(img)
#    if np.random.random() < th:
#        database[i, 12] = 0
#    else:
#        database[i, 12] = 1
#
#gs = np.zeros((2, 12), dtype=np.float32)
#f = np.zeros((2), dtype=np.float32)
#
#while True:
#    count = 0
#    for i in range(2):
#        gs[i] = np.mean(database[np.where(database[..., 12] == i)], axis=0)[0:12]
#
#    for i in range(len(file_list)):
#        for j in range(len(f)):
#            f[j] = np.sqrt(np.sum((gs[j] - database[i, 0:12])**2))
#        newclass = np.where(f == np.min(f))[0][0]
#        if database[i, 12] != newclass:
#            database[i, 12] = newclass
#            count += 1
#
#    #print(count)
#    #print(gs)
#    #print(database[..., 12])
#    if count == 0:
#        break
#
#for i in range(len(file_list)):
#    print(file_list[i], " Pred:", database[i, 12])
