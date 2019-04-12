import cv2
import numpy as np

def calc_hist(img, shade=4):
    data = np.zeros(shade*3)
    r = img[...,2]
    g = img[...,1]
    b = img[...,0]

    shade = 4
    for s in range(shade):
        data[s]         = np.sum(((s*64) <= b) & (b < ((s+1)*64)))
        data[s+shade]   = np.sum(((s*64) <= g) & (g < ((s+1)*64)))
        data[s+shade*2] = np.sum(((s*64) <= r) & (r < ((s+1)*64)))
    return data

np.random.seed(1)

database = np.zeros((4, 13), dtype=np.int)
class_name = ["akahara", "madara"]
file_list = ["test_akahara_1.jpg",
             "test_akahara_2.jpg",
             "test_madara_1.jpg",
             "test_madara_2.jpg"]

th = 0.5
for i in range(len(file_list)):
    img = cv2.imread("dataset/" + file_list[i])
    database[i, 0:12] = calc_hist(img)
    if np.random.random() < th:
        database[i, 12] = 0
    else:
        database[i, 12] = 1

gs = np.zeros((2, 12), dtype=np.float32)
f = np.zeros((2), dtype=np.float32)

count = 0
while True:
    for i in range(2):
        gs[i] = np.mean(database[np.where(database[..., 12] == i)], axis=0)[0:12]

    for i in range(len(file_list)):
        for j in range(len(f)):
            f[j] = np.sqrt(np.sum((gs[j] - database[i, 0:12])**2))
        newclass = np.where(f == np.min(f))[0][0]
        if database[i, 12] != newclass:
            database[i, 12] = newclass
            count += 1

    if count == 0:
        break

for i in range(len(file_list)):
    print(file_list[i], " Pred:", database[i, 12])
