import cv2
import numpy as np

T = 8
c_vector = np.ones((T, T), dtype=np.float)
c_vector[0, :] = 1 / np.sqrt(2)
c_vector[:, 0] = 1 / np.sqrt(2)
c_vector[0, 0] = (1 / np.sqrt(2)) ** 2

Q1 = np.array(((16, 11, 10, 16,  24,  40,  51,  61),
               (12, 12, 14, 19,  26,  58,  60,  55),
               (14, 13, 16, 24,  40,  57,  69,  56),
               (14, 17, 22, 29,  51,  87,  80,  62),
               (18, 22, 37, 56,  68, 109, 103,  77),
               (24, 35, 55, 64,  81, 104, 113,  92),
               (49, 64, 78, 87, 103, 121, 120, 101),
               (72, 92, 95, 98, 112, 100, 103,  99)), dtype=np.float32)

Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),
               (18, 21, 26, 66, 99, 99, 99, 99),
               (24, 26, 56, 99, 99, 99, 99, 99),
               (47, 66, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)

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
H, W, C = img.shape

R = img[..., 2]
G = img[..., 1]
B = img[..., 0]

Y = 0.299 * R + 0.5870 * G + 0.114 * B
Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128


T = 8
K = 8

X_Y  = np.zeros((H, W), dtype=np.float)
X_Cb = np.zeros((H, W), dtype=np.float)
X_Cr = np.zeros((H, W), dtype=np.float)

for i in range(H//T):
    for j in range(W//T):
        X_Y[(i*T):(i*T)+T, (j*T):(j*T)+T]  = np.round(DCT(Y[(i*T):(i*T)+T, (j*T):(j*T)+T])  / Q1) * Q1
        X_Cb[(i*T):(i*T)+T, (j*T):(j*T)+T] = np.round(DCT(Cb[(i*T):(i*T)+T, (j*T):(j*T)+T]) / Q2) * Q2
        X_Cr[(i*T):(i*T)+T, (j*T):(j*T)+T] = np.round(DCT(Cr[(i*T):(i*T)+T, (j*T):(j*T)+T]) / Q2) * Q2

for i in range(H//T):
    for j in range(W//T):
        Y[(i*T):(i*T)+T, (j*T):(j*T)+T]  = IDCT(X_Y[(i*T):(i*T)+T, (j*T):(j*T)+T])
        Cb[(i*T):(i*T)+T, (j*T):(j*T)+T] = IDCT(X_Cb[(i*T):(i*T)+T, (j*T):(j*T)+T])
        Cr[(i*T):(i*T)+T, (j*T):(j*T)+T] = IDCT(X_Cr[(i*T):(i*T)+T, (j*T):(j*T)+T])

R = Y + (Cr - 128) * 1.402
G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
B = Y + (Cb - 128) * 1.7718

out = np.zeros((H, W, C), dtype=np.float)
out[..., 2] = R
out[..., 1] = G
out[..., 0] = B

MSE = np.sum((img - out) ** 2) / (H * W)
PSNR = 10 * np.log10(255**2 / MSE)
bitrate = 8 * K**2 / 8**2

print(PSNR)
print(bitrate)

out[out < 0] = 0
out[out > 255] = 255

cv2.imwrite("my_answer_40.jpg", out.astype(np.uint8))
