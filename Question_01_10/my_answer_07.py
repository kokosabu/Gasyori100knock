import cv2
import numpy as np

img = cv2.imread("imori.jpg")
img = img.astype(np.float32)
size = 8

for i in range(len(img)//size):
    for j in range(len(img[i])//size):
        s = np.array([0.0, 0.0, 0.0])
        for k in range(size):
            for l in range(size):
                s += img[i*size+k, j*size+l]
        img[i*size:(i+1)*size, j*size:(j+1)*size] = s / (size*size)

img = img.astype(np.uint8)
cv2.imwrite("my_answer_07.jpg", img)
