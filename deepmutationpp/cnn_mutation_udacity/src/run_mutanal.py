import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from deepmutationpp.cnn_mutation_udacity.src.data_sort import check_data

import numpy as np

from properties import dataset_path

import datetime
print(datetime.datetime.now())

model_path = os.path.join('..', '..', 'original_models', 'udacity_original_0.h5')
mutants_path = os.path.join('..', '..', 'mutated_models_udacity')
results_path = os.path.join('..', '..', 'results_udacity')

# Strong TS

x_test = np.load(os.path.join('..', '..', dataset_path, 'Udacity', 'udacity_test_x.npy'))
y_test = np.load(os.path.join('..', '..',dataset_path, 'Udacity', 'udacity_test_y.npy'))

check_data(model_path, mutants_path, x_test, y_test, results_path, "test")

# Weak TS

x_test_weak = np.load(os.path.join('..', '..',dataset_path, 'Udacity', 'udacity_weak_test_x.npy'))
y_test_weak = np.load(os.path.join('..', '..',dataset_path, 'Udacity', 'udacity_weak_test_y.npy'))

check_data(model_path, mutants_path, x_test_weak, y_test_weak, results_path, "weak")


print(datetime.datetime.now())