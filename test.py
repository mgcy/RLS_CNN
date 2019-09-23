import numpy as np


def make_target_vector(label):
    # transfer labels to vectors
    target = np.zeros(10)
    target[label] = 1
    return target


def vect_gen_24(X, index):
    # X is the input image for convolution
    # index of the resulting image, from 0 to 575
    index_i = int(index / 24)
    index_j = index - index_i * 24
    print(index_i, index_j)
    tmp_mat = X[index_i:index_i + 5, index_j:index_j + 5]
    tmp_vect = tmp_mat.reshape((25))
    return tmp_vect


def vect_gen_8(X, index):
    # X is the input image for convolution
    # index of the resulting image, from 0 to 65
    index_i = int(index / 8)
    index_j = index - index_i * 8
    print(index_i, index_j)
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


def inverse_pool(S):
    # inverse pooling
    dim = int(S.shape[0])
    wid = int(S.shape[1] * 2)
    hei = int(S.shape[2] * 2)
    C = np.zeros((dim, wid, hei))
    print(C.shape)
    for i0 in range(dim):
        print(i0)
        for i1 in range(S.shape[1]):
            for i2 in range(S.shape[2]):
                C[i0, i1 * 2:i1 * 2 + 2, i2 * 2:i2 * 2 + 2] = S[i0, i1, i2]
    return C


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
    C = np.zeros((12, 4, 4))
    for i in range(12):
        C[i] = f[i * 16:(i + 1) * 16].reshape(4, 4)
    return C


def vect_to_image(s, index_i, index_j):
    # s with size 25
    im_s = s.reshape((5, 5))
    im = np.zeros((12, 12))
    im[index_i:index_i + 5, index_j:index_j + 5, ] = im_s
    return im


S = np.random.rand(12, 4, 4)
f = big_F(S)
back_S = F_inverse(f)
back_C = inverse_pool(back_S)

s = np.random.rand(25)
im = vect_to_image(s, 2, 3)
