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
for i in range(2):
    gs[i] = np.mean(database[np.where(database[..., 12] == i)], axis=0)[0:12]

print("assigned label")
print(database)
print("Grabity")
print(gs)

#insert_database(database, i, img, 0)
#img = cv2.imread("dataset/train_madara_"+str(i+1)+".jpg")
#insert_database(database, i+5, img, 1)
#
#correct = 0
#for i in range(2):
#    img = cv2.imread("dataset/test_akahara_"+str(i+1)+".jpg")
#    h = calc_hist(img)
#    f = np.zeros((10), dtype=np.int)
#    for j in range(10):
#        f[j] = np.sum(np.abs(h - database[j, 0:12]))
#    sf = sorted(f)
#    vote = np.zeros((2), dtype=np.int)
#    print("test_akahara_"+str(i+1)+".jpg is similar >> ", end="")
#    
#    for j in range(3):
#        idx = np.where(f == sf[j])[0][0]
#        vote[idx//5] += 1
#        print("train_"+class_name[idx//5]+"_"+str(idx%5+1)+".jpg, ", end="")
#
#    idx = np.where(vote == np.max(vote))[0][0]
#    if idx == 0:
#        correct += 1
#    print("| Pred >> "+class_name[idx])
#
#for i in range(2):
#    img = cv2.imread("dataset/test_madara_"+str(i+1)+".jpg")
#    h = calc_hist(img)
#    f = np.zeros((10), dtype=np.int)
#    for j in range(10):
#        f[j] = np.sum(np.abs(h - database[j, 0:12]))
#    sf = sorted(f)
#    vote = np.zeros((2), dtype=np.int)
#    print("test_madara_"+str(i+1)+".jpg is similar >> ", end="")
#    
#    for j in range(3):
#        idx = np.where(f == sf[j])[0][0]
#        vote[idx//5] += 1
#        print("train_"+class_name[idx//5]+"_"+str(idx%5+1)+".jpg, ", end="")
#
#    idx = np.where(vote == np.max(vote))[0][0]
#    if idx == 1:
#        correct += 1
#    print("| Pred >> "+class_name[idx])
#
#print("Accuracy >> "+str(correct/4)+" ("+str(correct)+"/"+str(4)+")")
