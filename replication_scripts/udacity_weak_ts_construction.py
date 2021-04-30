import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.config.experimental.set_visible_devices([], 'GPU')

import argparse

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd

from sklearn.utils import shuffle

from replication_scripts.batch_generator import Generator
from sklearn.model_selection import train_test_split

np.random.seed(0)


def get_label_buckets_udacity(y_train):
    std_array = np.std(y_train)

    index = 0
    bucket1 = []
    bucket1_ind = []
    bucket2 = []
    bucket2_ind = []
    bucket3 = []
    bucket3_ind = []

    for ind, element in enumerate(y_train):
        if element < -std_array:
            bucket1.append(element)
            bucket1_ind.append(ind)
        elif -std_array <= element <= std_array:
            bucket2.append(element)
            bucket2_ind.append(ind)
        elif element > std_array:
            bucket3.append(element)
            bucket3_ind.append(ind)

    buckets = [bucket1, bucket2, bucket3]
    buckets_ind = [bucket1_ind, bucket2_ind, bucket3_ind]

    unique_label_list = (0, 1, 2)
    unique_inverse = [-1] * len(y_train)
    unique_count = [-1] * len(unique_label_list)
    bucket_ind = 0
    for bucket in buckets:
        unique_count[bucket_ind] = len(bucket)
        for element in bucket:
            indices = np.argwhere(y_train == element)
            for index in indices:
                unique_inverse[index[0]] = bucket_ind

        bucket_ind = bucket_ind + 1
    return unique_label_list, np.asarray(unique_count), np.asarray(unique_inverse), buckets_ind


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    tracks = ["track1"]
    #drive = ["normal", "reverse", "recovery", "recovery2", "sport_normal"]
    drive = ['normal', 'recovery', 'reverse']

    x = None
    y = None
    path = None
    x_train = None
    y_train = None
    x_valid = None
    y_valid = None

    for track in tracks:
        for drive_style in drive:
            try:
                path = os.path.join('datasets', args.data_dir, track, drive_style, 'driving_log.csv')
                data_df = pd.read_csv(path)
                if x is None:
                    x = data_df[['center', 'left', 'right']].values
                    y = data_df['steering'].values
                else:
                    x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
                    y = np.concatenate((y, data_df['steering'].values), axis=0)
            except FileNotFoundError:
                print("Unable to read file %s" % path)
                continue

    if x is None or y is None:
        print("No driving data were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    try:
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=args.test_size, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    print("Train dataset: " + str(len(x_train)) + " elements")
    print("Test dataset: " + str(len(x_valid)) + " elements")
    return x_train, x_valid, y_train, y_valid


def get_generators(args, x_train, x_valid, y_train, y_valid):
    # shuffle the data because they are sequential; should help over-fitting towards certain parts of the track only
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_valid, y_valid = shuffle(x_valid, y_valid, random_state=0)

    x_train: 'x_train'
    y_train: 'y_train'

    # data for training are augmented, data for validation are not
    train_generator = Generator(x_train, y_train, True, args)
    validation_generator = Generator(x_valid, y_valid, False, args)

    return train_generator, validation_generator


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def get_udacity_weak_ts(model_dir, dataset_dir, weak_dataset_dir):

    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str,
                        default=dataset_dir)
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=50)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=100)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=64)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)

    args = parser.parse_args()

    data = load_data(args)
    train_generator, validation_generator = get_generators(args, *data)

    validation_generator_list = list(validation_generator)

    x_test = []
    y_true = []

    for i in range(len(validation_generator_list)):
        for j in range(args.batch_size):
            # idx = i * args.batch_size + j
            x_test.append(validation_generator_list[i][0][j])
            y_true.append(validation_generator_list[i][1][j])

    x_test = np.array(x_test)
    y_test = np.array(y_true)

    list_of_losses = []

    for i in range(0, 20):
        model_file = os.path.join(model_dir, 'udacity_original_' + str(i) + '.h5')

        model = tf.keras.models.load_model(model_file)

        prediction_simp = model.predict(x_test)

        pred_un = []
        for kk in range(len(prediction_simp)):
            pred_un.append(prediction_simp[kk][0])

        loss = [abs(yp - yt) for yp, yt in zip(pred_un, y_true)]


        list_of_losses.append(loss)

    std_of_losses = np.std(list_of_losses, axis=0)

    unique_label_list, unique_counts, unique_inverse, buckets = get_label_buckets_udacity(y_true)

    print(unique_label_list, unique_counts, unique_inverse)

    buckets = [buckets[1]]

    buckets_size_weak = 1000

    chosen_elements = []

    for i, bucket in enumerate(buckets):
        bucket_loss = std_of_losses[bucket]
        sorted_by_loss = np.argsort(bucket_loss)

        sorted_by_loss_subset = sorted_by_loss[-buckets_size_weak:]

        bucket = np.asarray(bucket)
        chosen_subset = bucket[sorted_by_loss_subset]
        chosen_elements.extend(chosen_subset)

    x_test_sub = x_test[chosen_elements]
    y_test_sub = y_test[chosen_elements]

    np.save(os.path.join(weak_dataset_dir, 'udacity_weak_test_x.npy'), x_test_sub)
    np.save(os.path.join(weak_dataset_dir, 'udacity_weak_test_y.npy'), y_test_sub)


if __name__ == '__main__':

    get_udacity_weak_ts(model_dir = os.path.join(), #'/Users/nhumbatova/Documents/UDACITY/udacity_original/'
                        dataset_dir = os.path.join(), #'/Users/nhumbatova/Documents/UDACITY/datasets/'
                        weak_dataset_dir = os.path.join()) # '/Users/nhumbatova/Documents/UDACITY/datasets/weak/'
