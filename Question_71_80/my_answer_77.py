import cv2
import numpy as np

K = 111
s = 10
g = 1.2
l = 10
p = 0
A = 0

def x_dash(x, y):
    return np.cos(A*np.pi/180) * x + np.sin(A*np.pi/180) * y

def y_dash(x, y):
    return -np.sin(A*np.pi/180) * x + np.cos(A*np.pi/180) * y

def G(y, x):
    return np.exp(-(x_dash(x, y)**2 + g**2 * y_dash(x, y)**2) / (2 * s**2)) * np.cos(2 * np.pi * x_dash(x, y) / l + p)

filt = np.zeros((K, K))
for y in range(K):
    for x in range(K):
        filt[y, x] = G(y-K//2, x-K//2)

filt = (filt - np.min(filt)) * (1 / (np.max(filt) - np.min(filt)))
filt *= 255
cv2.imwrite("my_answer_77.jpg", filt.astype(np.uint8))
