import os
import numpy as np
import tensorflow as tf
import datetime
tf.config.experimental.set_visible_devices([], 'GPU')
from sklearn.model_selection import train_test_split
from deepmutationpp.cnn_mutation_lenet.src.data_sort import check_data

from properties import dataset_path


print(datetime.datetime.now())

model_path = os.path.join('..', '..', 'original_models', 'lenet_original_0.h5')
mutants_path = os.path.join('..', '..', 'mutated_models_lenet')
results_path = os.path.join('..', '..', 'results_lenet')

x_img = np.load(os.path.join('..', '..', dataset_path, 'UnityEyes', 'dataset_x_img.npy'))
x_head_angles = np.load(os.path.join('..', '..', dataset_path, 'UnityEyes', 'dataset_x_head_angles_np.npy'))
y_gaze_angles = np.load(os.path.join('..', '..', dataset_path, 'UnityEyes', 'dataset_y_gaze_angles_np.npy'))

x_img_train, x_img_test, x_ha_train, x_ha_test, y_gaze_train, y_gaze_test = train_test_split(x_img,
                                                                                             x_head_angles,
                                                                                             y_gaze_angles,
                                                                                             test_size=0.2,
                                                                                             random_state=42)

x_test = [x_img_test, x_ha_test]
y_test = y_gaze_test

check_data(model_path, mutants_path, x_test, y_test, results_path, "test")

# Weak
x_img_test_weak = np.load(os.path.join('..', '..', dataset_path, 'UnityEyes', 'srch_img_10.npy'))
x_ha_test_weak = np.load(os.path.join('..', '..', dataset_path, 'UnityEyes', 'srch_ha_10.npy'))
y_gaze_test_weak = np.load(os.path.join('..', '..', dataset_path, 'UnityEyes', 'srch_gaze_10.npy'))

x_test_weak = [x_img_test_weak, x_ha_test_weak]
y_test_weak = y_gaze_test_weak

check_data(model_path, mutants_path, x_test_weak, y_test_weak, results_path, "weak")


print(datetime.datetime.now())