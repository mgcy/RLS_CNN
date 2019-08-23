import numpy as np
import matplotlib.pyplot as plt

B_len = 200
mat_dim = 25
mse = []


def inverse_thm(A, B, B_len):
    for i in range(B_len):
        if i == 0:
            g_1 = 1 / (1 + np.trace(np.matmul(A, B[0])))
            C = A - g_1 * np.matmul(np.matmul(A, B[0]), A)
        else:
            g_2 = 1 / (1 + np.trace(np.matmul(C, B[i])))
            C = C - g_2 * np.matmul(np.matmul(C, B[i]), C)
    return C


A = np.identity(mat_dim)

for i1 in range(1, B_len):
    print(i1)
    B = np.zeros([i1, mat_dim, mat_dim])
    A_inverse = np.linalg.inv(A)
    T = A
    for i in range(i1):
        tmp = np.random.rand(mat_dim, 1)
        # tmp = np.ones([3, 1])
        B[i] = tmp * np.transpose(tmp)
        T = T + B[i]
    T = T - A

    R = inverse_thm(A, B, i1)
    # print('Via Algorithm:')
    # print(R)
    T_inverse = np.linalg.inv(T)
    # print('Actual: ')
    # print(T_inverse)
    mse.append(((R - T_inverse) ** 2).mean(axis=None))

plt.plot(mse)
plt.ylabel('MSE Element-Wisely')
plt.xlabel('# of Samples')
plt.ylim((0, 1))
plt.show()
