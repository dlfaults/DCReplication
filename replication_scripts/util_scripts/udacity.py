import os
import numpy as np

from properties import dataset_path


def get_test_set(dataset_type):

    if dataset_type == 'T':
        y_test = np.load(os.path.join(dataset_path, 'Udacity', 'udacity_test_y.npy'))

        return y_test
    else:
        y_test_weak = np.load(os.path.join(dataset_path, 'Udacity', 'udacity_weak_test_y.npy'))

        return y_test_weak