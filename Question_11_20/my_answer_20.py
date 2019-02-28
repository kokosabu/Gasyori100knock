import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("imori_dark.jpg")

plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("my_answer_20.png")
plt.show()
