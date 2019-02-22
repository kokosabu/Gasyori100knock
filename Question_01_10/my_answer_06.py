import cv2
import numpy as np

img = cv2.imread("imori.jpg")

r = img[:,:,2]
g = img[:,:,1]
b = img[:,:,0]

r[np.where((  0 <= r) & (r <  64))] =  32
r[np.where(( 64 <= r) & (r < 128))] =  96
r[np.where((128 <= r) & (r < 192))] = 160
r[np.where((192 <= r) & (r < 256))] = 224
g[np.where((  0 <= g) & (g <  64))] =  32
g[np.where(( 64 <= g) & (g < 128))] =  96
g[np.where((128 <= g) & (g < 192))] = 160
g[np.where((192 <= g) & (g < 256))] = 224
b[np.where((  0 <= b) & (b <  64))] =  32
b[np.where(( 64 <= b) & (b < 128))] =  96
b[np.where((128 <= b) & (b < 192))] = 160
b[np.where((192 <= b) & (b < 256))] = 224

img[:,:,2] = r
img[:,:,1] = g
img[:,:,0] = b

cv2.imwrite("my_answer_06.jpg", img)
