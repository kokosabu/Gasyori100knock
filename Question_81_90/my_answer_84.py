import cv2
import numpy as np
import matplotlib.pyplot as plt

def insert_database(database, i, img, c):
    r = img[...,2]
    g = img[...,1]
    b = img[...,0]

    shade = 4
    for s in range(shade):
        database[i, s] = np.sum(((s*64) <= b) & (b < ((s+1)*64)))
        database[i, s+shade] = np.sum(((s*64) <= g) & (g < ((s+1)*64)))
        database[i, s+shade*2] = np.sum(((s*64) <= r) & (r < ((s+1)*64)))
    database[i, shade*3] = c

database = np.zeros((10, 13), dtype=np.int)

left = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
fig, axes = plt.subplots(2, 5)
for i in range(5):
    img = cv2.imread("dataset/train_akahara_"+str(i+1)+".jpg")
    insert_database(database, i, img, 0)
    axes[0][i].set_title("train_akahara_"+str(i+1)+".jpg")
    axes[0][i].bar(left, database[i, 0:12])
    img = cv2.imread("dataset/train_madara_"+str(i+1)+".jpg")
    insert_database(database, i+5, img, 1)
    axes[1][i].set_title("train_madara_"+str(i+1)+".jpg")
    axes[1][i].bar(left, database[i+5, 0:12])

print(database)
plt.savefig("my_answer_84.png")
