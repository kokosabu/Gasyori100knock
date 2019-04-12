import cv2
import numpy as np

def calc_hist(img, shade=4):
    data = np.zeros(shade*3)
    r = img[...,2]
    g = img[...,1]
    b = img[...,0]

    shade = 4
    for s in range(shade):
        data[s] = np.sum(((s*64) <= b) & (b < ((s+1)*64)))
        data[s+shade] = np.sum(((s*64) <= g) & (g < ((s+1)*64)))
        data[s+shade*2] = np.sum(((s*64) <= r) & (r < ((s+1)*64)))
    return data

def insert_database(database, i, img, c):
    database[i, 0:12] = calc_hist(img)
    database[i, 12] = c

database = np.zeros((10, 13), dtype=np.int)
class_name = ["akahara", "madara"]

for i in range(5):
    img = cv2.imread("dataset/train_akahara_"+str(i+1)+".jpg")
    insert_database(database, i, img, 0)
    #file_database
    img = cv2.imread("dataset/train_madara_"+str(i+1)+".jpg")
    insert_database(database, i+5, img, 1)

for i in range(2):
    img = cv2.imread("dataset/test_akahara_"+str(i+1)+".jpg")
    h = calc_hist(img)
    f = np.zeros((10), dtype=np.int)
    for j in range(10):
        f[j] = np.sum(np.abs(h - database[j, 0:12]))
    idx = np.where(f == np.min(f))[0][0]
    print("test_akahara_"+str(i+1)+".jpg is similar >> train_"+class_name[idx//5]+"_"+str(idx%5+1)+".jpg  Pred >> "+class_name[idx//5])

for i in range(2):
    img = cv2.imread("dataset/test_madara_"+str(i+1)+".jpg")
    h = calc_hist(img)
    f = np.zeros((10), dtype=np.int)
    for j in range(10):
        f[j] = np.sum(np.abs(h - database[j, 0:12]))
    idx = np.where(f == np.min(f))[0][0]
    print("test_akahara_"+str(i+1)+".jpg is similar >> train_"+class_name[idx//5]+"_"+str(idx%5+1)+".jpg  Pred >> "+class_name[idx//5])
