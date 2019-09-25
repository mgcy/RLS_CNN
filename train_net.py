from keras.datasets import mnist
import numpy as np


def inverse_thm_fn(A, B, B_len):
    for i in range(B_len):
        if i == 0:
            g_1 = 1 / (1 + np.trace(np.matmul(A, B[0])))
            C = A - g_1 * np.matmul(np.matmul(A, B[0]), A)
        else:
            g_2 = 1 / (1 + np.trace(np.matmul(C, B[i])))
            C = C - g_2 * np.matmul(np.matmul(C, B[i]), C)
    return C


def make_target_vector(label, max_value):
    target = np.ones(10) * 0.5
    target[label] = max_value
    return target


def vect_gen(X, index_i, index_j):
    # X is the input image for convolution
    # index of the resulting image, from 0 to 23
    tmp_mat = X[index_i:index_i + 5, index_j:index_j + 5]
    tmp_vect = tmp_mat.reshape(25)
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


def inverse_pool(S):
    # inverse pooling
    dim = int(S.shape[0])
    wid = int(S.shape[1] * 2)
    hei = int(S.shape[2] * 2)
    C = np.zeros((dim, wid, hei))
    for i0 in range(dim):
        for i1 in range(S.shape[1]):
            for i2 in range(S.shape[2]):
                C[i0, i1 * 2:i1 * 2 + 2, i2 * 2:i2 * 2 + 2] = S[i0, i1, i2]
    return C


def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


def diff_sigmoid(x):
    diff = sigmoid(x) * (1 - sigmoid(x))
    return diff


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


def get_Y(S1, r, lp, betap, k2):
    # r from 0 to 24^2-1
    Y = 0
    for l in range(6):
        if l != lp:
            index_i = int(r / 8)
            index_j = r - index_i * 8
            Y = Y + np.matmul(vect_gen(S1, index_i, index_j), k2[l, betap, :])
    return Y


def get_delta2(r, net2, P_C2):
    index_i = int(r / 8)
    index_j = r - index_i * 8
    net2_r = net2[index_i, index_j]
    e2_r = P_C2[index_i, index_j] * diff_sigmoid(net2_r)
    delta2 = net2_r + e2_r
    return delta2


def vect_to_image(s, index_i, index_j):
    # s with size 25
    im_s = s.reshape((5, 5))
    im = np.zeros((12, 12))
    im[index_i:index_i + 5, index_j:index_j + 5, ] = im_s
    return im


###############################  START   #################################
# Load the dataset
nb_samples = 5
max_value = 5

(X_train, Y_train), (_, _) = mnist.load_data()

width = X_train.shape[1]
height = X_train.shape[2]

X = X_train[0:nb_samples].astype(np.float64) / 255.0
Y = Y_train[0:nb_samples]
Y_tar = np.zeros((nb_samples, 10))  # 10
for i in range(nb_samples):
    Y_tar[i] = make_target_vector(Y[i], max_value)

# initialization parameters
k1 = np.random.uniform(-1, 1, (6, 25))  # 6 x 25
k2 = np.random.uniform(-1, 1, (6, 12, 25))  # 6 x 12 x 25
W = np.random.uniform(-1, 1, (192, 10))  # 192 x 10

############################### training   #################################
for t in range(nb_samples):
    # temp results
    C1 = np.zeros((6, 24, 24))  # 6 x 24 x 24
    C2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
    S1 = np.zeros((6, 12, 12))  # 6 x 12 x12
    S2 = np.zeros((12, 4, 4))  # 12 x 4 x 4
    net1 = np.zeros((6, 24, 24))  # 6 x 24 x24
    net2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
    R1 = np.zeros((6, 25, 25))
    R2 = np.zeros((6, 12, 25, 25))
    for i1 in range(6):
        R1[i1] = np.identity(25)
        for i2 in range(12):
            R2[i1, i2] = np.identity(25)
    # first convolution and pooling
    for f1 in range(6):
        for i1 in range(24):
            for j1 in range(24):
                x = vect_gen(X[0], i1, j1)
                net1[f1, i1, j1] = np.dot(x, k1[f1])
        C1[f1] = sigmoid(net1[f1])
        S1[f1] = max_pool(C1[f1])
        # second convolution
        for i2 in range(8):
            for j2 in range(8):
                s1 = vect_gen(S1[f1], i2, j2)
                for f2 in range(12):
                    net2[f2, i2, j2] = net2[f2, i2, j2] + np.dot(s1, k2[f1, f2])
    # second pooling
    for f2 in range(12):
        C2[f2] = sigmoid(net2[f2])
        S2[f2] = max_pool(C2[f2])

    # F operation
    f = big_F(S2)  # 192

    # fully connected layer
    net3 = sigmoid(np.matmul(f, W))  # 10

    # RLS training
    f = f.reshape(192, 1)
    ft = f.reshape(1, 192)
    # obtain R3
    R3 = np.matmul(f, f.reshape(1, 192))  # 192 x 192
    for i in range(10):
        # obtain Delta3
        delta3 = f * Y_tar[t, i]
        # update W
        P = np.identity(192)
        g = (np.matmul(P, f)) / (1 + np.matmul(np.matmul(ft, P), f))
        P = np.matmul((np.identity(192) - np.matmul(g, ft)), P)
        W[:, i] = W[:, i] + np.matmul(g, (Y_tar[t, i] - np.matmul(ft, W[:, i])))

    # obtain partial L / partial f
    P_f = np.zeros(192)
    for q in range(10):
        e3 = Y_tar[t, q] - np.matmul(f.reshape(192), W[:, q])
        P_f = P_f - W[:, q] * e3

    # obtain partial L / parital S2
    P_S2 = F_inverse(P_f)  # 12 x 4 x 4

    # obtain partial L / parial C2
    P_C2 = inverse_pool(P_S2)  # 12 x 8 x 8

    # update k2
    for lp in range(6):
        for betap in range(12):
            # obtain Delta2
            Delta2 = 0
            B2 = np.zeros((64, 25, 25))
            for r in range(64):
                index_i = int(r / 8)
                index_j = r - index_i * 8
                # obtain delta2
                delta2 = get_delta2(r, net2[betap], P_C2[betap])
                y_r = get_Y(S1[lp], r, lp, betap, k2)
                s1 = vect_gen(S1[lp], index_i, index_j)
                Delta2 = Delta2 + s1 * (delta2 - y_r)
                B2[r] = np.outer(s1, s1)
            D2 = inverse_thm_fn(R2[lp, betap], B2, 64)  # D2 = inv(R2)
            k2[lp, betap] = np.matmul(D2, Delta2)
            # print('k2_' + str(lp) + '_' + str(betap) + ' updated!')

    # Before updating k1, re-obtain net2
    for f1 in range(6):
        for i2 in range(8):
            for j2 in range(8):
                s1 = vect_gen(S1[f1], i2, j2)
                for f2 in range(12):
                    net2[f2, i2, j2] = net2[f2, i2, j2] + np.dot(s1, k2[f1, f2])

    # obtain partial L / parital S1
    P_S1 = np.zeros((6, 12, 12))  # 6 x 12 x 12
    for alphap in range(6):
        for m1 in range(8):
            for m2 in range(8):
                P_s1 = np.zeros(25)  # 25 x 1
                for u in range(12):
                    P_s1 = P_s1 + P_C2[u, m1, m2] * diff_sigmoid(net2[u, m1, m2]) * k2[alphap, u]
                P_S1[alphap] = P_S1[alphap] + vect_to_image(P_s1, m1, m2)

    # obtain partial L / parial C1
    P_C1 = inverse_pool(P_S1)  # 6 x 24 x 24

    # update k1
    for alphap in range(6):
        Delta1 = np.zeros(25)
        B1 = np.zeros((576, 25, 25))
        for y1 in range(24):
            for y2 in range(24):
                e1 = P_C1[alphap, y1, y2] + diff_sigmoid(net1[alphap, y1, y2])
                delta1 = net1[alphap, y1, y2] + e1
                x0 = vect_gen(X[t], y1, y2)  # change X[0]!
                Delta1 = Delta1 + x0 * delta1
                B1[y1 * 24 + y2] = np.outer(x0, x0)
        D1 = inverse_thm_fn(R1[alphap], B1, 576)  # D1= inv(R1)
        k1[alphap] = np.matmul(D1, Delta1)

#########################    Testing     ##################
'''
# temp results
C1 = np.zeros((6, 24, 24))  # 6 x 24 x 24
C2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
S1 = np.zeros((6, 12, 12))  # 6 x 12 x12
S2 = np.zeros((12, 4, 4))  # 12 x 4 x 4
net1 = np.zeros((6, 24, 24))  # 6 x 24 x24
net2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
accuracy = 0
# make testing data
test_samples = 10
X = X_train[nb_samples:nb_samples + test_samples].astype(np.float64) / 255.0
Y = Y_train[nb_samples:nb_samples + test_samples]
Y_tar = np.zeros((nb_samples, 10))
for i in range(nb_samples):
    Y_tar[i] = make_target_vector(Y[i], max_value)  # 10
Y_hat = np.zeros(test_samples)
for t in range(test_samples):
    # first convolution and pooling
    for f1 in range(6):
        for i1 in range(24):
            for j1 in range(24):
                x = vect_gen(X[t], i1, j1)
                net1[f1, i1, j1] = np.dot(x, k1[f1])
        C1[f1] = sigmoid(net1[f1])
        S1[f1] = max_pool(C1[f1])
        # second convolution
        for i2 in range(8):
            for j2 in range(8):
                s1 = vect_gen(S1[f1], i2, j2)
                for f2 in range(12):
                    net2[f2, i2, j2] = net2[f2, i2, j2] + np.dot(s1, k2[f1, f2])
    # second pooling
    for f2 in range(12):
        C2[f2] = sigmoid(net2[f2])
        S2[f2] = max_pool(C2[f2])

    # F operation
    f = big_F(S2)  # 192

    # fully connected layer
    net3 = sigmoid(np.matmul(f, W))  # 10

    # count accuracy
    Y_hat[t] = np.argmax(net3)
    if int(Y_hat[t]) == int(Y[t]):
        accuracy = accuracy + 1

acc_precent = accuracy / test_samples
print(acc_precent)
'''