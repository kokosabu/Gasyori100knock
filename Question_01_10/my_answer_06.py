import cv2
import numpy as np

img = cv2.imread("imori.jpg")

r = img[:,:,2]
g = img[:,:,1]
b = img[:,:,0]

r[np.where((  0 <= r) &  (r <  63))] =  32
r[np.where(( 64 <= r) &  (r < 127))] =  96
r[np.where((127 <= r) &  (r < 191))] = 160
r[np.where((191 <= r) &  (r < 256))] = 224
g[np.where((  0 <= g) &  (g <  63))] =  32
g[np.where(( 64 <= g) &  (g < 127))] =  96
g[np.where((127 <= g) &  (g < 191))] = 160
g[np.where((191 <= g) &  (g < 256))] = 224
b[np.where((  0 <= b) &  (b <  63))] =  32
b[np.where(( 64 <= b) &  (b < 127))] =  96
b[np.where((127 <= b) &  (b < 191))] = 160
b[np.where((191 <= b) &  (b < 256))] = 224

img[:,:,2] = r
img[:,:,1] = g
img[:,:,0] = b

cv2.imwrite("my_answer_06.jpg", img)
