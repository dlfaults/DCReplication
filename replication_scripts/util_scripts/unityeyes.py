import os
import numpy as np

from sklearn.model_selection import train_test_split

from properties import dataset_path

def get_test_set(dataset_type):

    if dataset_type == 'T':
        y_gaze_angles = np.load(os.path.join(dataset_path, 'UnityEyes', 'dataset_y_gaze_angles_np.npy'))

        y_gaze_train, y_gaze_test = train_test_split(y_gaze_angles,test_size=0.2,random_state=42)

        y_test = y_gaze_test

        return y_test
    else:

        y_gaze_test_weak = np.load(os.path.join(dataset_path, 'UnityEyes', 'srch_gaze_10.npy'))

        y_test_weak = y_gaze_test_weak

        return y_test_weak