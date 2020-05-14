from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import confusion_matrix


# from sklearn.metrics import accuracy_score

def inverse_thm_fn(A, B, B_len):
    g_1 = 1 / (1 + np.trace(np.matmul(A, B[0])))
    D = A - g_1 * np.matmul(np.matmul(A, B[0]), A)
    for i_thm in range(1, B_len):
        g_2 = 1 / (1 + np.trace(np.matmul(D, B[i_thm])))
        D = D - g_2 * np.matmul(np.matmul(D, B[i_thm]), D)
    return D


def make_target_vector(label, max_target_value):
    # linear
    # target = np.zeros(10)
    # target[label] = 1
    # non-linear
    target = np.ones(10) * 0.5
    target[label] = max_target_value
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
    # concatenate 12 4-by-4 images, return num_k2 * 16-D f vector
    len_S = S.shape[0]  # 12
    wid_S = S.shape[1]  # 4
    hei_S = S.shape[2]  # 4
    f_F = np.zeros(1)
    for i_F in range(len_S):
        f_F = np.concatenate((f_F, S[i_F].reshape((wid_S * hei_S))))
    f_F = np.delete(f_F, 0)

    return f_F


def F_inverse(f, num_k2):
    # len_f = len(f) # 12
    S = np.zeros((num_k2, 4, 4))
    for i_F_inv in range(num_k2):
        S[i_F_inv] = f[i_F_inv * 16:(i_F_inv + 1) * 16].reshape(4, 4)
    return S


def get_Y(S1_Y, r_i, r_j, lpY, betapY, k2Y, num_k1):
    # r from 0 to 8^2-1
    Y_value = 0
    for ly in range(num_k1):
        if ly != lpY:
            Y_value = Y_value + np.matmul(np.concatenate((vect_gen(S1_Y[ly], r_i, r_j), [1])), k2Y[ly, betapY])
    return Y_value


def get_delta2(net2, P_C2):
    # index_i2 = int(r2 / 8)
    # index_j2 = r2 - index_i2 * 8
    # net2_r = net2_delta[index_i2, index_j2]
    # e2_r = P_C2_delta2[index_i2, index_j2] * diff_sigmoid(net2_r)
    # get_delta2_v = net2_r + e2_r
    tmp_get_delta2 = P_C2 * diff_sigmoid(net2) + net2
    return tmp_get_delta2


def vect_to_image(s, index_i_im, index_j_im):
    # s with size 25
    im_s = s.reshape((5, 5))
    im = np.zeros((12, 12))
    im[index_i_im:index_i_im + 5, index_j_im:index_j_im + 5] = im_s
    return im

    ###############################  START   #################################
    # def main():


# monitor start time
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
print('Start Training...')
# Load the dataset
train_samples = 6000
test_samples = 1000
max_value = 5
forget_factor = 0.99
train_iter = 20

np.random.seed(3)

# number of filters in each layer
num_k1 = 6  # 6 before
num_k2 = 12  # 12 before

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# width = X_train.shape[1]  # 28
# height = X_train.shape[2]  # 28

X_train = X_train[0:train_samples].astype(np.float64) / 255.0
Y_train = Y_train[0:train_samples]
Y_tar = np.zeros((train_samples, 10))  # 10
for i in range(train_samples):
    Y_tar[i] = make_target_vector(Y_train[i], max_value)

# initialization parameters, "+1" for bias
k1 = np.random.uniform(-0.1, 0.1, (num_k1, 25 + 1))  # 6 x 25
k2 = np.random.uniform(-0.1, 0.1, (num_k1, num_k2, 25 + 1))  # 6 x 12 x 25
W = np.random.uniform(-0.1, 0.1, (num_k2 * 16 + 1, 10))  # num_k2 * 16 x 10

Jcost = np.zeros(train_iter)  # cost of every iteration
# Jcost = np.zeros(train_iter * train_samples)
############################### training   #################################
for iteration in range(train_iter):
    print('Iteration: ' + str(iteration))

    R1 = np.zeros((num_k1, 25 + 1, 25 + 1))
    R2 = np.zeros((num_k1, num_k2, 25 + 1, 25 + 1))
    for i1 in range(num_k1):  # initialize R1 and R2
        R1[i1] = 100 * np.identity(25 + 1)
        for i2 in range(num_k2):
            R2[i1, i2] = 100 * np.identity(25 + 1)

    Delta1 = np.zeros((num_k1, 25 + 1))
    Delta2 = np.zeros((num_k1, num_k2, 25 + 1))
    P = np.zeros((10, num_k2 * 16 + 1, num_k2 * 16 + 1))
    for i in range(10):
        P[i] = np.identity(num_k2 * 16 + 1)

    Lcost = 0  # cost of every iter
    # Jcost = np.zeros(train_samples)
    # start T training samples
    for t in range(train_samples):
        # temp results
        C1 = np.zeros((num_k1, 24, 24))  # 6 x 24 x 24
        C2 = np.zeros((num_k2, 8, 8))  # 12 x 8 x 8
        S1 = np.zeros((num_k1, 12, 12))  # 6 x 12 x12
        S2 = np.zeros((num_k2, 4, 4))  # 12 x 4 x 4
        net1 = np.zeros((num_k1, 24, 24))  # 6 x 24 x24
        net2 = np.zeros((num_k2, 8, 8))  # 12 x 8 x 8

        # first convolution and pooling
        for f1 in range(num_k1):
            for i1 in range(24):
                for j1 in range(24):
                    x = vect_gen(X_train[t], i1, j1)
                    x = np.concatenate((x, [1]))  # add bias
                    net1[f1, i1, j1] = np.dot(x, k1[f1])
            C1[f1] = sigmoid(net1[f1])
            S1[f1] = mean_pool(C1[f1])
            # second convolution
            for i2 in range(8):
                for j2 in range(8):
                    s1 = vect_gen(S1[f1], i2, j2)
                    s1 = np.concatenate((s1, [1]))  # add bias
                    for f2 in range(num_k2):
                        net2[f2, i2, j2] = net2[f2, i2, j2] + np.dot(s1, k2[f1, f2])
        # second pooling
        for f2 in range(num_k2):
            C2[f2] = sigmoid(net2[f2])
            S2[f2] = mean_pool(C2[f2])

        # F operation
        f = big_F(S2)  # num_k2 * 16
        f = np.concatenate((f, [1]))  # add bias, num_k2 * 16+1
        # fully connected layer
        net3 = sigmoid(np.matmul(f, W))  # 10

        # obtain cost
        Lcost = Lcost + forget_factor * np.linalg.norm(Y_tar[t] - net3)
        # Jcost[t] = np.linalg.norm(Y_tar[t] - net3)
        # RLS training
        f = f.reshape(num_k2 * 16 + 1, 1)
        ft = f.reshape(1, num_k2 * 16 + 1)
        # obtain R3
        # R3 = np.matmul(f, f.reshape(1, num_k2 * 16))  # num_k2 * 16 x num_k2 * 16

        P_f = np.zeros(num_k2 * 16 + 1)

        for i in range(10):
            # obtain Delta3
            # delta3 = f * Y_tar[t, i]
            # update W
            g = (np.matmul(P[i], f)) / (forget_factor + np.matmul(np.matmul(ft, P[i]), f))
            P[i] = 1 / forget_factor * np.matmul((np.identity(num_k2 * 16 + 1) - np.matmul(g, ft)), P[i])
            W[:, i] = W[:, i] + np.matmul(g, (Y_tar[t, i] - np.matmul(ft, W[:, i])))
            # modified:
            # obtain partial L / partial f
            e3 = np.matmul(f.reshape(num_k2 * 16 + 1), W[:, i]) - Y_tar[t, i]
            P_f = P_f + W[:, i] * e3

        '''
        # obtain partial L / partial f
        P_f = np.zeros(num_k2 * 16)
        for q in range(10):
            e3 = np.matmul(f.reshape(num_k2 * 16), W[:, q]) - Y_tar[t, q]
            P_f = P_f + W[:, q] * e3
        '''

        # obtain partial L / parital S2
        P_S2 = F_inverse(P_f[0:num_k2 * 16], num_k2)  # 12 x 4 x 4

        # obtain partial L / parial C2
        P_C2 = inverse_pool(P_S2)  # 12 x 8 x 8
        P_S1 = np.zeros((num_k1, 12, 12))  # 6 x 12 x 12, initialize partial L / partial S1

        # update k2
        delta2 = get_delta2(net2, P_C2)  # obtain delta2, new
        for lp in range(num_k1):
            P_s1 = np.zeros(25)  # 25 x 1, for partial L / partial S1
            for betap in range(num_k2):
                tmp_Delta2 = np.zeros(25 + 1)
                B2 = np.zeros((8, 8, 25 + 1, 25 + 1))
                for r_i in range(8):
                    for r_j in range(8):
                        # index_i = int(r / 8)
                        # index_j = r - index_i * 8
                        # obtain delta2 old
                        # delta2 = get_delta2(r, net2[betap], P_C2[betap])
                        y_r = get_Y(S1, r_i, r_j, lp, betap, k2, num_k1)
                        s1 = vect_gen(S1[lp], r_i, r_j)
                        s1 = np.concatenate((s1, [1]))  # add bias
                        # Delta2[lp, betap] = forget_factor * Delta2[lp, betap] + s1 * (y_r - delta2)
                        tmp_Delta2 = tmp_Delta2 + s1 * (delta2[betap, r_i, r_j] - y_r)
                        B2[r_i, r_j] = np.outer(s1, s1)
                        # obtain partial L / partial S1 inside loop
                        P_s1 = P_s1 + P_C2[betap, r_i, r_j] * diff_sigmoid(net2[betap, r_i, r_j]) * k2[lp, betap, 0:25]
                        P_S1[lp] = P_S1[lp] + vect_to_image(P_s1, r_i, r_j)
                B2 = np.reshape(B2, (64, 25 + 1, 25 + 1))
                Delta2[lp, betap] = forget_factor * Delta2[lp, betap] + tmp_Delta2
                R2[lp, betap] = inverse_thm_fn((1 / forget_factor) * R2[lp, betap], B2, 64)  # D2 = inv(R2)
                k2[lp, betap] = np.matmul(R2[lp, betap], Delta2[lp, betap])  # update k2
                # two different ways to obtain partial L / partial S1, depending on using old k2 or new k2
                # opinion: need to used the updated k2
                # obtain partial L / partial S1 outside loop
                # P_s1 = P_s1 + P_C2[betap, r_i, r_j] * diff_sigmoid(net2[betap, r_i, r_j]) * k2[lp, betap, 0:25]
                # P_S1[lp] = P_S1[lp] + vect_to_image(P_s1, r_i, r_j)
                '''
                # Before updating k1, re-obtain net2, but why?
                # net2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
                net2[betap] = np.zeros((8, 8))
                for f1 in range(num_k1):
                    for i2 in range(8):
                        for j2 in range(8):
                            s1 = vect_gen(S1[f1], i2, j2)
                            s1 = np.concatenate((s1, [1]))  # add bias
                            # net2[betap] = np.zeros((8, 8))
                            # for f2 in range(12):
                            net2[betap, i2, j2] = net2[betap, i2, j2] + np.dot(s1, k2[f1, betap])
                '''
                # print('k2_' + str(lp) + '_' + str(betap) + ' updated!')
        '''
        # Before updating k1, re-obtain net2
        net2 = np.zeros((12, 8, 8))  # 12 x 8 x 8
        for f1 in range(6):
            for i2 in range(8):
                for j2 in range(8):
                    s1 = vect_gen(S1[f1], i2, j2)
                    for f2 in range(12):
                        net2[f2, i2, j2] = net2[f2, i2, j2] + np.dot(s1, k2[f1, f2])
        '''
        '''
        # obtain partial L / partial S1
        P_S1 = np.zeros((num_k1, 12, 12))  # 6 x 12 x 12
        for alphap in range(num_k1):
            for m1 in range(8):
                for m2 in range(8):
                    P_s1 = np.zeros(25)  # 25 x 1
                    for u in range(num_k2):
                        P_s1 = P_s1 + P_C2[u, m1, m2] * diff_sigmoid(net2[u, m1, m2]) * k2[alphap, u, 0:25]
                    P_S1[alphap] = P_S1[alphap] + vect_to_image(P_s1, m1, m2)
        '''
        # obtain partial L / parial C1
        P_C1 = inverse_pool(P_S1)  # 6 x 24 x 24

        # update k1
        # e1 = P_C1 * diff_sigmoid(net1)
        delta1 = net1 + P_C1 * diff_sigmoid(net1)
        for alphap in range(num_k1):
            # Delta1 = np.zeros(25)
            tmp_Delta1 = np.zeros(25 + 1)
            B1 = np.zeros((576, 25 + 1, 25 + 1))
            for y1 in range(24):
                for y2 in range(24):
                    # e1 = P_C1[alphap, y1, y2] * diff_sigmoid(net1[alphap, y1, y2])
                    # delta1 = net1[alphap, y1, y2] + e1
                    x0 = vect_gen(X_train[t], y1, y2)  #
                    x0 = np.concatenate((x0, [1]))  # add bias
                    tmp_Delta1 = tmp_Delta1 + x0 * delta1[alphap, y1, y2]
                    # Delta1[alphap] = forget_factor * Delta1[alphap] + x0 * delta1
                    B1[y1 * 24 + y2] = np.outer(x0, x0)
            Delta1[alphap] = forget_factor * Delta1[alphap] + tmp_Delta1
            R1[alphap] = inverse_thm_fn((1 / forget_factor) * R1[alphap], B1, 576)  # D1= inv(R1)
            k1[alphap] = np.matmul(R1[alphap], Delta1[alphap])

        ##########################################################################
        '''
        # obtain cost
        C1 = np.zeros((num_k1, 24, 24))  # 6 x 24 x 24
        C2 = np.zeros((num_k2, 8, 8))  # 12 x 8 x 8
        S1 = np.zeros((num_k1, 12, 12))  # 6 x 12 x12
        S2 = np.zeros((num_k2, 4, 4))  # 12 x 4 x 4
        net1 = np.zeros((num_k1, 24, 24))  # 6 x 24 x24
        net2 = np.zeros((num_k2, 8, 8))  # 12 x 8 x 8

        # first convolution and pooling
        for f1 in range(num_k1):
            for i1 in range(24):
                for j1 in range(24):
                    x = vect_gen(X_train[t], i1, j1)
                    x = np.concatenate((x, [1]))  # add bias
                    net1[f1, i1, j1] = np.dot(x, k1[f1])
            C1[f1] = sigmoid(net1[f1])
            S1[f1] = max_pool(C1[f1])
            # second convolution
            for i2 in range(8):
                for j2 in range(8):
                    s1 = vect_gen(S1[f1], i2, j2)
                    s1 = np.concatenate((s1, [1]))  # add bias
                    for f2 in range(num_k2):
                        net2[f2, i2, j2] = net2[f2, i2, j2] + np.dot(s1, k2[f1, f2])
        # second pooling
        for f2 in range(num_k2):
            C2[f2] = sigmoid(net2[f2])
            S2[f2] = max_pool(C2[f2])

        # F operation
        f = big_F(S2)  # num_k2 * 16
        f = np.concatenate((f, [1]))  # add bias, num_k2 * 16+1

        # fully connected layer
        net3 = np.matmul(f, W)  # 10
        Lcost = Lcost + forget_factor * np.linalg.norm(Y_tar[t] - net3)
        '''
    Jcost[iteration] = Lcost
    # monitor progress
    # if (t + 1) % (train_samples / 10) == 0:
    #     print(str(int((t + 1) / train_samples * 100)) + '%...')
    # plt.plot(Jcost)
    # plt.show()

#########################    Testing     ##################
print('Training Finished! Start Testing...')
# temp results
accuracy = 0
# make testing data

# X_test = X_train[train_samples:train_samples + test_samples].astype(np.float64) / 255.0
X_test = X_test[0:test_samples].astype(np.float64) / 255.0
# Y_test = Y_train[train_samples:train_samples + test_samples]
Y_test = Y_test[0:test_samples]
Y_hat_test = np.zeros(test_samples)
for t1 in range(test_samples):
    C1 = np.zeros((num_k1, 24, 24))  # 6 x 24 x 24
    C2 = np.zeros((num_k2, 8, 8))  # 12 x 8 x 8
    S1 = np.zeros((num_k1, 12, 12))  # 6 x 12 x12
    S2 = np.zeros((num_k2, 4, 4))  # 12 x 4 x 4
    net1 = np.zeros((num_k1, 24, 24))  # 6 x 24 x24
    net2 = np.zeros((num_k2, 8, 8))  # 12 x 8 x 8
    # first convolution and pooling
    for f1 in range(num_k1):
        for i1 in range(24):
            for j1 in range(24):
                x = vect_gen(X_test[t1], i1, j1)
                x = np.concatenate((x, [1]))  # add bias
                net1[f1, i1, j1] = np.dot(x, k1[f1])
        C1[f1] = sigmoid(net1[f1])
        S1[f1] = mean_pool(C1[f1])
        # second convolution
        for i2 in range(8):
            for j2 in range(8):
                s1 = vect_gen(S1[f1], i2, j2)
                s1 = np.concatenate((s1, [1]))  # add bias
                for f2 in range(num_k2):
                    net2[f2, i2, j2] = net2[f2, i2, j2] + np.dot(s1, k2[f1, f2])
    # second pooling
    C2 = sigmoid(net2)
    for f2 in range(num_k2):
        S2[f2] = mean_pool(C2[f2])

    # F operation
    f = big_F(S2)  # num_k2 * 16
    f = np.concatenate((f, [1]))  # add bias, num_k2 * 16+1

    # fully connected layer
    net3 = sigmoid(np.matmul(f, W))  # 10

    # count accuracy
    Y_hat_test[t1] = np.argmax(net3)
    if int(Y_hat_test[t1]) == int(Y_test[t1]):
        accuracy = accuracy + 1

acc_precent = accuracy / test_samples
print("Accuracy:")
print(acc_precent)
Y_test.astype(int)
Y_hat_test.astype(int)
confusion_m = confusion_matrix(Y_test, Y_hat_test)  # confusion matrix
print("Confusion Matrix")
print(confusion_m)

# monitor end time
end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time = end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))
plt.plot(Jcost)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('plot.png')

# if __name__ == '__main__':
# print("START")
# main()
