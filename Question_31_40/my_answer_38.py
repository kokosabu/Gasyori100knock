import cv2
import numpy as np

T = 8
c_vector = np.ones((T, T), dtype=np.float)
c_vector[0, :] = 1 / np.sqrt(2)
c_vector[:, 0] = 1 / np.sqrt(2)
c_vector[0, 0] = (1 / np.sqrt(2)) ** 2

Q = np.array(((16, 11, 10, 16,  24,  40,  51,  61),
              (12, 12, 14, 19,  26,  58,  60,  55),
              (14, 13, 16, 24,  40,  57,  69,  56),
              (14, 17, 22, 29,  51,  87,  80,  62),
              (18, 22, 37, 56,  68, 109, 103,  77),
              (24, 35, 55, 64,  81, 104, 113,  92),
              (49, 64, 78, 87, 103, 121, 120, 101),
              (72, 92, 95, 98, 112, 100, 103,  99)), dtype=np.float32)

def DCT(f, K=8):
    F = np.zeros((T, T), dtype=np.float)
    for v in range(T):
        y_v = np.cos((np.arange(T).repeat(T).reshape(T, -1) * 2 + 1) * v * np.pi / (2*T))
        for u in range(T):
            x_v = np.cos((np.tile(np.arange(T), (T, 1)) * 2 + 1) * u * np.pi / (2*T))
            F[v, u] = np.sum(f * x_v * y_v)
    return F * c_vector * 2 / T

def IDCT(F, K=8):
    cv = c_vector[0:K, 0:K]
    f = np.zeros((T, T), dtype=np.float)
    for y in range(T):
        v_v = np.cos((2 * y + 1) * np.arange(K).repeat(K).reshape(K, -1) * np.pi / (2*T))
        for x in range(T):
            u_v = np.cos((2 * x + 1) * np.tile(np.arange(K), (K, 1)) * np.pi / (2*T))
            f[y, x] = np.sum(cv * F[0:K, 0:K] * u_v * v_v)
    return f * 2 / T

img = cv2.imread("imori.jpg").astype(np.float)
H, W, _ = img.shape

gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

out = np.zeros((H, W), dtype=np.float)

T = 8
for i in range(H//T):
    for j in range(W//T):
        out[(i*T):(i*T)+T, (j*T):(j*T)+T] = np.round(DCT(gray[(i*T):(i*T)+T, (j*T):(j*T)+T]) / Q) * Q

for i in range(H//T):
    for j in range(W//T):
        out[(i*T):(i*T)+T, (j*T):(j*T)+T] = IDCT(out[(i*T):(i*T)+T, (j*T):(j*T)+T])

K = 8
MSE = np.sum((gray - out) ** 2) / (H * W)
PSNR = 10 * np.log10(255**2 / MSE)
bitrate = 8 * K**2 / 8**2

print(PSNR)
print(bitrate)

out[out < 0] = 0
out[out > 255] = 255

cv2.imwrite("my_answer_38.jpg", out.astype(np.uint8))
