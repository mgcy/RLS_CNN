from keras.datasets import mnist
import numpy as np

def make_target_vector(label):
    target = np.zeros(10)
    target[label] = 1
    return target
# Load the dataset

nb_samples = 1

(X_train, Y_train), (_, _) = mnist.load_data()

width = X_train.shape[1]
height = X_train.shape[2]

X = X_train[0:nb_samples].reshape((nb_samples, width * height)).astype(np.float32) / 255.0
Y = Y_train[0:nb_samples]
Y_tar = make_target_vector(Y)


