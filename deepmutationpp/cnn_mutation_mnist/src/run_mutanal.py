import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import h5py
import datetime
from keras.datasets import mnist

from keras import backend as K

from deepmutationpp.cnn_mutation_mnist.src.data_sort import check_data

from properties import dataset_path

print(datetime.datetime.now())

model_path = os.path.join('..', '..', 'original_models', 'mnist_original_0.h5')
mutants_path = os.path.join('..', '..', 'mutated_models_mnist')
results_path = os.path.join('..', '..', 'results_mnist')

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows, img_cols = 28, 28
num_classes = 10

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255
# print(x_test.shape[0], 'test samples')

check_data(model_path, mutants_path, x_test, y_test, results_path, "test")

# Weak
hf = h5py.File(os.path.join('..', '..', dataset_path, 'MNIST', 'dataset_easy.h5'), 'r')
x_test_weak = np.asarray(hf.get('x_test'))
y_test_weak = np.asarray(hf.get('y_test'))

check_data(model_path, mutants_path, x_test_weak, y_test_weak, results_path, "weak")

print(datetime.datetime.now())