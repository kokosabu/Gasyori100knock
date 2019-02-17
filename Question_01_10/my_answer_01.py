import cv2

img = cv2.imread("imori.jpg")
img[:, :] = img[:, :, (2, 1, 0)]
cv2.imwrite("my_answer_01.jpg", img)

#cv2.imshow('', img)
#cv2.waitKey(0)
