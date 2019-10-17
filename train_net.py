from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


def inverse_thm_fn(A, B, B_len):
    for i_thm in range(B_len):
        if i_thm == 0:
            g_1 = 1 / (1 + np.trace(np.matmul(A, B[0])))
            D = A - g_1 * np.matmul(np.matmul(A, B[0]), A)
        else:
            g_2 = 1 / (1 + np.trace(np.matmul(D, B[i_thm])))
            D = D - g_2 * np.matmul(np.matmul(D, B[i_thm]), D)
    return D


def make_target_vector(label, max_target_value):
    # linear
    target = np.zeros(10)
    target[label] = 1
    # non-linear
    # target = np.ones(10) * 0.5
    # target[label] = max_target_value
    return target


def vect_gen(X_gen, index_i_gen, index_j_gen):
    # X is the input image for convolution with 5 x 5 filter size
    tmp_mat = X_gen[index_i_gen:index_i_gen + 5, index_j_gen:index_j_gen + 5]
    tmp_vect = tmp_mat.reshape(25)
    return tmp_vect


def max_pool(C):
    # max pooling with strip 2
    wid = int(C.shape[0] / 2)
    hei = int(C.shape[1] / 2)
    S = np.zeros((wid, hei))
    for i1_max in range(wid):
        for i2_max in range(hei):
            S[i1_max, i2_max] = np.amax(C[i1_max * 2:i1_max * 2 + 2, i2_max * 2:i2_max * 2 + 2])
    return S


def mean_pool(C):
    # avg pooling with strip 2
    wid = int(C.shape[0] / 2)
    hei = int(C.shape[1] / 2)
    S = np.zeros((wid, hei))
    for i1_mean in range(wid):
        for i2_mean in range(hei):
            S[i1_mean, i2_mean] = np.mean(C[i1_mean * 2:i1_mean * 2 + 2, i2_mean * 2:i2_mean * 2 + 2])
    return S


def inverse_pool(S):
    # inverse pooling
    dim = int(S.shape[0])
    wid = int(S.shape[1] * 2)
    hei = int(S.shape[2] * 2)
    C = np.zeros((dim, wid, hei))
    for i0 in range(dim):
        for i1_inv in range(S.shape[1]):
            for i2_inv in range(S.shape[2]):
                C[i0, i1_inv * 2:i1_inv * 2 + 2, i2_inv * 2:i2_inv * 2 + 2] = S[i0, i1_inv, i2_inv]
    return C


def sigmoid(x_v):
    sigm = 1. / (1. + np.exp(-x_v))
    return sigm


def diff_sigmoid(x_d):
    diff = sigmoid(x_d) * (1 - sigmoid(x_d))
    return diff


def big_F(S):
    # concatenate 12 4-by-4 images, return 192-D f vector
    len_S = S.shape[0]  # 12
    wid_S = S.shape[1]  # 4
    hei_S = S.shape[2]  # 4
    f_F = np.zeros(1)
    for i_F in range(len_S):
        f_F = np.concatenate((f_F, S[i_F].reshape((wid_S * hei_S))))
    f_F = np.delete(f_F, 0)

    return f_F


def F_inverse(f):
    # len_f = len(f) # 12
    S = np.zeros((12, 4, 4))
    for i_F_inv in range(12):
        S[i] = f[i_F_inv * 16:(i_F_inv + 1) * 16].reshape(4, 4)
    return S


def get_Y(S1_Y, r_Y, lpY, betapY, k2Y):
    # r from 0 to 8^2-1
    Y_value = 0
    for ly in range(6):
        if ly != lpY:
            get_Y_i = int(r / 8)
            get_Y_j = r_Y - get_Y_i * 8
            Y_value = Y_value + np.matmul(vect_gen(S1_Y[ly], get_Y_i, get_Y_j), k2Y[ly, betapY])
    return Y_value


def get_delta2(r2, net2_delta, P_C2_delta2):
    index_i2 = int(r2 / 8)
    index_j2 = r2 - index_i2 * 8
    net2_r = net2_delta[index_i2, index_j2]
    e2_r = P_C2_delta2[index_i2, index_j2] * diff_sigmoid(net2_r)
    get_delta2_v = net2_r + e2_r
    return get_delta2_v, e2_r


def vect_to_image(s, index_i_im, index_j_im):
    # s with size 25
    im_s = s.reshape((5, 5))
    im = np.zeros((12, 12))
    im[index_i_im:index_i_im + 5, index_j_im:index_j_im + 5] = im_s
    return im


###############################  START   #################################
print('Start Training...')
# Load the dataset
nb_samples = 6000
max_value = 5
forget_factor = 0.99
train_iter = 100

(X_train, Y_train), (_, _) = mnist.load_data()

width = X_train.shape[1]  # 28
height = X_train.shape[2]  # 28

X = X_train[0:nb_samples].astype(np.float64) / 255.0
Y = Y_train[0:nb_samples]
Y_tar = np.zeros((nb_samples, 10))  # 10
for i in range(nb_samples):
    Y_tar[i] = make_target_vector(Y[i], max_value)

# initialization parameters
k1 = np.random.uniform(-1, 1, (6, 25))  # 6 x 25
k2 = np.random.uniform(-1, 1, (6, 12, 25))  # 6 x 12 x 25
W = np.random.uniform(-1, 1, (192, 10))  # 192 x 10

Jcost = np.zeros(train_iter)  # cost of every iteration

############################### training   #################################
for iteration in range(train_iter):
    R1 = np.zeros((6, 25, 25))
    R2 = np.zeros((6, 12, 25, 25))
    for i1 in range(6):  # initialize R1 and R2
        R1[i1] = np.identity(25)
        for i2 in range(12):
            R2[i1, i2] = np.identity(25)

    Delta1 = np.zeros((6, 25))
    Delta2 = np.zeros((6, 12, 25))
    P = np.zeros((10, 192, 192))
    for i in range(10):
        P[i] = np.identity(192)

    Lcost = 0  # cost of every image

    # start T training samples
    for t in range(nb_samples):
        # temp results
        C1 = np.zeros((6, 24, 24))  # 6 x 24 x 24
        C2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
        S1 = np.zeros((6, 12, 12))  # 6 x 12 x12
        S2 = np.zeros((12, 4, 4))  # 12 x 4 x 4
        net1 = np.zeros((6, 24, 24))  # 6 x 24 x24
        net2 = np.zeros((12, 8, 8))  # 12 x 8 x 8

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
        # net3 = sigmoid(np.matmul(f, W))  # 10

        # RLS training
        f = f.reshape(192, 1)
        ft = f.reshape(1, 192)
        # obtain R3
        # R3 = np.matmul(f, f.reshape(1, 192))  # 192 x 192
        for i in range(10):
            # obtain Delta3
            # delta3 = f * Y_tar[t, i]
            # update W
            g = (np.matmul(P[i], f)) / (forget_factor + np.matmul(np.matmul(ft, P[i]), f))
            P[i] = 1 / forget_factor * np.matmul((np.identity(192) - np.matmul(g, ft)), P[i])
            W[:, i] = W[:, i] + np.matmul(g, (Y_tar[t, i] - np.matmul(ft, W[:, i])))

        # obtain partial L / partial f
        P_f = np.zeros(192)
        for q in range(10):
            e3 = np.matmul(f.reshape(192), W[:, q]) - Y_tar[t, q]
            P_f = P_f + W[:, q] * e3

        # obtain partial L / parital S2
        P_S2 = F_inverse(P_f)  # 12 x 4 x 4

        # obtain partial L / parial C2
        P_C2 = inverse_pool(P_S2)  # 12 x 8 x 8

        # update k2
        for lp in range(6):
            for betap in range(12):
                tmp_Delta2 = np.zeros(25)
                B2 = np.zeros((64, 25, 25))
                for r in range(64):
                    index_i = int(r / 8)
                    index_j = r - index_i * 8
                    # obtain delta2
                    delta2, deltae2 = get_delta2(r, net2[betap], P_C2[betap])
                    y_r = get_Y(S1, r, lp, betap, k2)
                    s1 = vect_gen(S1[lp], index_i, index_j)
                    # Delta2[lp, betap] = forget_factor * Delta2[lp, betap] + s1 * (y_r - delta2)
                    tmp_Delta2 = tmp_Delta2 + s1 * (delta2 - y_r)
                    B2[r] = np.outer(s1, s1)
                Delta2[lp, betap] = forget_factor * Delta2[lp, betap] + tmp_Delta2
                R2[lp, betap] = inverse_thm_fn((1 / forget_factor) * R2[lp, betap], B2, 64)  # D2 = inv(R2)
                k2[lp, betap] = np.matmul(R2[lp, betap], Delta2[lp, betap])
                # Before updating k1, re-obtain net2
                net2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
                for f1 in range(6):
                    for i2 in range(8):
                        for j2 in range(8):
                            s1 = vect_gen(S1[f1], i2, j2)
                            for f2 in range(12):
                                net2[f2, i2, j2] = net2[f2, i2, j2] + np.dot(s1, k2[f1, f2])

                # print('k2_' + str(lp) + '_' + str(betap) + ' updated!')

        # Before updating k1, re-obtain net2
        net2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
        for f1 in range(6):
            for i2 in range(8):
                for j2 in range(8):
                    s1 = vect_gen(S1[f1], i2, j2)
                    for f2 in range(12):
                        net2[f2, i2, j2] = net2[f2, i2, j2] + np.dot(s1, k2[f1, f2])

        # obtain partial L / partial S1
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
            # Delta1 = np.zeros(25)
            tmp_Delta1 = np.zeros(25)
            B1 = np.zeros((576, 25, 25))
            for y1 in range(24):
                for y2 in range(24):
                    e1 = P_C1[alphap, y1, y2] + diff_sigmoid(net1[alphap, y1, y2])
                    delta1 = net1[alphap, y1, y2] + e1
                    x0 = vect_gen(X[t], y1, y2)  # change X[0]!
                    tmp_Delta1 = tmp_Delta1 + x0 * delta1
                    # Delta1[alphap] = forget_factor * Delta1[alphap] + x0 * delta1
                    B1[y1 * 24 + y2] = np.outer(x0, x0)
            Delta1[alphap] = forget_factor * Delta1[alphap] + tmp_Delta1
            R1[alphap] = inverse_thm_fn((1 / forget_factor) * R1[alphap], B1, 576)  # D1= inv(R1)
            k1[alphap] = np.matmul(R1[alphap], Delta1[alphap])
            # update net1
            for f1 in range(6):
                for i1 in range(24):
                    for j1 in range(24):
                        x = vect_gen(X[t], i1, j1)
                        net1[f1, i1, j1] = np.dot(x, k1[f1])

        ##########################################################################
        # obtain cost
        C1 = np.zeros((6, 24, 24))  # 6 x 24 x 24
        C2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
        S1 = np.zeros((6, 12, 12))  # 6 x 12 x12
        S2 = np.zeros((12, 4, 4))  # 12 x 4 x 4
        net1 = np.zeros((6, 24, 24))  # 6 x 24 x24
        net2 = np.zeros((12, 8, 8))  # 12 x 8 x 8

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
        net3 = np.matmul(f, W)  # 10
        Lcost = Lcost + forget_factor * np.linalg.norm(Y_tar[t] - net3)
    Jcost[iteration] = Lcost
    # monitor progress
    if (iteration + 1) % (train_iter / 20) == 0:
        print(str(int((iteration + 1) / train_iter * 100)) + '%...')
plt.plot(Jcost)
plt.show()
'''
#########################    Testing     ##################
print('Training Finished! Start Testing...')
# temp results
accuracy = 0
# make testing data
test_samples = 1000
X = X_train[nb_samples:nb_samples + test_samples].astype(np.float64) / 255.0
Y = Y_train[nb_samples:nb_samples + test_samples]
Y_hat = np.zeros(test_samples)
for t in range(test_samples):
    C1 = np.zeros((6, 24, 24))  # 6 x 24 x 24
    C2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
    S1 = np.zeros((6, 12, 12))  # 6 x 12 x12
    S2 = np.zeros((12, 4, 4))  # 12 x 4 x 4
    net1 = np.zeros((6, 24, 24))  # 6 x 24 x24
    net2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
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
