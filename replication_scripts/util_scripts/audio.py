import os
import numpy as np

from properties import dataset_path

SAMPLING_RATE = 16000


def get_test_df(type):
    if type == 'T':
        strong_y = np.load(os.path.join(dataset_path, "Audio", "audio_labels_test.npy"))

        return strong_y
    else:
        easy_y = np.load(os.path.join(dataset_path, "Audio", "audio_labels_weak.npy"))

        return easy_y
