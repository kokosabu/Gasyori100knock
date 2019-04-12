import numpy as np

def iou(a, b):
    start_x = np.max((a[0], b[0]))
    start_y = np.max((a[1], b[1]))
    end_x = np.min((a[2], b[2]))
    end_y = np.min((a[3], b[3]))
    w = end_x - start_x
    h = end_y - start_y
    if w < 0:
        w = 0
    if h < 0:
        h = 0

    R1 = (a[2]-a[0]) * (a[3]-a[1])
    R2 = (b[2]-b[0]) * (b[3]-b[1])
    Rol = w * h

    return Rol / (R1 + R2 - Rol)

a = np.array((50, 50, 150, 150), dtype=np.float32)
b = np.array((60, 60, 170, 160), dtype=np.float32)

print(iou(a, b))
