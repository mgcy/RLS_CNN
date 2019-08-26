from keras.datasets import mnist
import numpy as np


def make_target_vector(label, max_value):
    target = np.ones(10) * 0.5
    target[label] = max_value
    return target


def vect_gen(X, index_i, index_j):
    # X is the input image for convolution
    # index of the resulting image, from 0 to 23
    tmp_mat = X[index_i:index_i + 5, index_j:index_j + 5]
    tmp_vect = tmp_mat.reshape((25))
    return tmp_vect


def max_pool(C):
    # max pooling with strip 2
    wid = int(C.shape[0] / 2)
    hei = int(C.shape[1] / 2)
    S = np.zeros((wid, hei))
    for i1 in range(wid):
        for i2 in range(hei):
            S[i1, i2] = np.amax(C[i1 * 2:i1 * 2 + 2, i2 * 2:i2 * 2 + 2])
    return S


def mean_pool(C):
    # avg pooling with strip 2
    wid = int(C.shape[0] / 2)
    hei = int(C.shape[1] / 2)
    S = np.zeros((wid, hei))
    for i1 in range(wid):
        for i2 in range(hei):
            S[i1, i2] = np.mean(C[i1 * 2:i1 * 2 + 2, i2 * 2:i2 * 2 + 2])
    return S


def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


def big_F(S):
    # concatenate 12 4-by-4 images, return 192-D f vector
    len_S = S.shape[0]  # 12
    wid_S = S.shape[1]  # 4
    hei_S = S.shape[2]  # 4
    f = np.zeros((1))
    for i in range(len_S):
        f = np.concatenate((f, S[i].reshape((wid_S * hei_S))))
    f = np.delete(f, 0)

    return f


def F_inverse(f):
    # len_f = len(f) # 12
    S = np.zeros((12, 4, 4))
    for i in range(12):
        S[i] = f[i * 16:(i + 1) * 16].reshape(4, 4)
    return S


# Load the dataset

# Try the single-image case first
nb_samples = 1
max_value = 50

(X_train, Y_train), (_, _) = mnist.load_data()

width = X_train.shape[1]
height = X_train.shape[2]

X = X_train[0:nb_samples].astype(np.float32) / 255.0
Y = Y_train[0:nb_samples]
Y_tar = make_target_vector(Y, max_value)

# initialization
# parameters
k1 = np.random.uniform(-1, 1, (6, 25))
k2 = np.random.uniform(-1, 1, (6, 12, 25))
W = np.random.uniform(-1, 1, (192, 10))
# temp results
C1 = np.zeros((6, 24, 24))
C2 = np.zeros((12, 8, 8))
S1 = np.zeros((6, 12, 12))
S2 = np.zeros((12, 4, 4))

# first convolution and pooling
for f1 in range(6):
    for i1 in range(24):
        for j1 in range(24):
            x = vect_gen(X[0], i1, j1)
            C1[f1, i1, j1] = sigmoid(np.dot(x, k1[f1]))
    S1[f1] = max_pool(C1[f1])
    # second convolution
    for i2 in range(8):
        for j2 in range(8):
            s1 = vect_gen(S1[f1], i2, j2)
            for f2 in range(12):
                C2[f2, i2, j2] = C2[f2, i2, j2] + np.dot(s1, k2[f1, f2])
# second pooling
for f2 in range(12):
    C2[f2] = sigmoid(C2[f2])
    S2[f2] = max_pool(C2[f2])

# F operation
f = big_F(S2)

# fully connected layer
net = sigmoid(np.matmul(f, W))

# RLS training
f = f.reshape(192, 1)
ft = f.reshape(1, 192)
R1 = np.matmul(f, f.reshape(1, 192))
for i in range(10):
    Delta1 = f * Y_tar[i]
    # update W
    P = np.identity(192)
    g = (np.matmul(P, f)) / (1 + np.matmul(np.matmul(ft, P), f))
    P = np.matmul((np.identity(192) - np.matmul(g, ft)), P)
    W[:, i] = W[:, i] + np.matmul(g, (Y_tar[i] - np.matmul(ft, W[:, i])))

# update k2
# obtain partial L / partial f
P_f = np.zeros(192)
for i in range(10):
    P_f = P_f - W[:, i] * e3
