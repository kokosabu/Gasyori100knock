import cv2
import numpy as np

img = cv2.imread("imori.jpg")

size = 8

for i in range(len(img)//size):
    for j in range(len(img[i])//size):
        img[i*size:(i+1)*size, j*size:(j+1)*size, 2] = np.max(img[i*size:(i+1)*size, j*size:(j+1)*size, 2]).astype(np.int)
        img[i*size:(i+1)*size, j*size:(j+1)*size, 1] = np.max(img[i*size:(i+1)*size, j*size:(j+1)*size, 1]).astype(np.int)
        img[i*size:(i+1)*size, j*size:(j+1)*size, 0] = np.max(img[i*size:(i+1)*size, j*size:(j+1)*size, 0]).astype(np.int)

cv2.imwrite("my_answer_08.jpg", img)
