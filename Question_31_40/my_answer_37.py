import cv2
import numpy as np

T = 8
c_vector = np.ones((T, T), dtype=np.float)
c_vector[0, :] = 1 / np.sqrt(2)
c_vector[:, 0] = 1 / np.sqrt(2)
c_vector[0, 0] = (1 / np.sqrt(2)) ** 2

# http://www.enjoy.ne.jp/~k-ichikawa/DCTran.html
def DCT(f, K=8):
    F = np.zeros((T, T), dtype=np.float)
    for v in range(T):
        y_v = np.cos((np.arange(T).repeat(T).reshape(T, -1) * 2 + 1) * v * np.pi / (2*T))
        for u in range(T):
            x_v = np.cos((np.tile(np.arange(T), (T, 1)) * 2 + 1) * u * np.pi / (2*T))
            F[v, u] = np.sum(f * x_v * y_v)
    return F * c_vector

def IDCT(F, K=8):
    cv = c_vector[0:K, 0:K]
    f = np.zeros((T, T), dtype=np.float)
    for y in range(T):
        v_v = np.cos((2 * y + 1) * np.arange(K).repeat(K).reshape(K, -1) * np.pi / (2*T))
        for x in range(T):
            u_v = np.cos((2 * x + 1) * np.tile(np.arange(K), (K, 1)) * np.pi / (2*T))
            f[y, x] = np.sum(cv * F[0:K, 0:K] * u_v * v_v)
    return f * 4 / (T * T)

img = cv2.imread("imori.jpg").astype(np.float)
H, W, _ = img.shape

gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

out = np.zeros((H, W), dtype=np.float)

T = 8
for i in range(H//T):
    for j in range(W//T):
        out[(i*T):(i*T)+T, (j*T):(j*T)+T] = DCT(gray[(i*T):(i*T)+T, (j*T):(j*T)+T])

for i in range(H//T):
    for j in range(W//T):
        out[(i*T):(i*T)+T, (j*T):(j*T)+T] = IDCT(out[(i*T):(i*T)+T, (j*T):(j*T)+T], K=4)

K = 4
MSE = np.sum((gray - out) ** 2) / (H * W)
PSNR = 10 * np.log10(255**2 / MSE)
bitrate = 8 * K**2 / 8**2
print(PSNR)
print(bitrate)

out[out < 0] = 0
out[out > 255] = 255

cv2.imwrite("my_answer_37.jpg", out.astype(np.uint8))
