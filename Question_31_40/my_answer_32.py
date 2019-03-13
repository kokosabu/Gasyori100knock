import cv2
import numpy as np

# https://ja.wikipedia.org/wiki/%E9%9B%A2%E6%95%A3%E3%83%95%E3%83%BC%E3%83%AA%E3%82%A8%E5%A4%89%E6%8F%9B#2%E6%AC%A1%E5%85%83%E3%81%A7%E3%81%AE%E5%A4%89%E6%8F%9B
def dft(img, N):
    W = np.zeros((N, N), dtype=np.complex)
    for y in range(N):
        for x in range(N):
            W[y, x] = np.exp(-2j * np.pi * y * x / N)
    return np.matmul(np.matmul(W, img), W)

def inv_dft(img, N):
    W = np.zeros((N, N), dtype=np.complex)
    for y in range(N):
        for x in range(N):
            W[y, x] = np.exp(2j * np.pi * y * x / N)
    return np.matmul(np.matmul(W, img), W) / (N*N)

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

out = np.zeros((H, W, C), dtype=np.float)

out = dft(gray, H)
out2 = inv_dft(out, H)
out = (np.abs(out) / np.abs(out).max() * 255).astype(np.uint8)
out2 = np.abs(out2)

out[out < 0] = 0
out[out > 255] = 255
out2[out2 < 0] = 0
out2[out2 > 255] = 255

cv2.imwrite("my_answer_32_ps.jpg", out.astype(np.uint8))
cv2.imwrite("my_answer_32.jpg", out2.astype(np.uint8))
