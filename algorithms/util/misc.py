import numpy as np


def safe_divide(arr1, arr2):
    out = []
    a1 = arr1.flatten()
    l1 = len(a1)
    a2 = arr2.flatten()
    l2 = len(a2)
    m = int(l1 / l2)
    a3 = np.repeat(a2, m)

    for i in range(len(a1)):
        if a3[i] == 0:
            out.append(0)
        else:
            out.append(a1[i] / a3[i])
    return np.array(out).reshape(arr1.shape)

