import os

from keras.datasets import mnist

import numpy as np
import h5py

from properties import dataset_path

def get_test_set(dataset_type):

    if dataset_type == 'T':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        return y_test
    else:
        file_path = os.path.join(dataset_path, "MNIST", 'dataset_easy.h5')
        hf = h5py.File(file_path, 'r')

        # x_test_weak = np.asarray(hf.get('x_test'))
        y_test_weak = np.asarray(hf.get('y_test'))

        return y_test_weak