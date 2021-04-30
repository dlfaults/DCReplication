import csv
import os
import numpy as np
import pandas as pd
import glob

import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
from tensorflow import math

from replication_scripts.util_scripts.audio import get_test_df as audio_ts
from replication_scripts.util_scripts.unityeyes import get_test_set as unityeyes_ts
from replication_scripts.util_scripts.mnist import get_test_set as mnist_ts
from replication_scripts.util_scripts.udacity import get_test_set as udacity_ts

from properties import data_path


def angle_loss_fn_batch(y_true, y_pred):
    x_p = math.sin(y_pred[:, 0]) * math.cos(y_pred[:, 1])
    y_p = math.sin(y_pred[:, 0]) * math.sin(y_pred[:, 1])
    z_p = math.cos(y_pred[:, 0])

    x_t = math.sin(y_true[:, 0]) * math.cos(y_true[:, 1])
    y_t = math.sin(y_true[:, 0]) * math.sin(y_true[:, 1])
    z_t = math.cos(y_true[:, 0])

    norm_p = math.sqrt(x_p * x_p + y_p * y_p + z_p * z_p)
    norm_t = math.sqrt(x_t * x_t + y_t * y_t + z_t * z_t)

    dot_pt = x_p * x_t + y_p * y_t + z_p * z_t

    angle_value = dot_pt/(norm_p * norm_t)
    angle_value = tf.clip_by_value(angle_value, -0.99999, 0.99999)

    loss_val = (math.acos(angle_value))

    return loss_val


def calc_ms_audio(dataset, dataset_type):

    dataset = list(dataset.as_numpy_iterator())

    y = []
    # x = []
    for i, ds in enumerate(dataset):
        # x.extend(ds[0])
        y.extend(ds[1])

    # x = np.asarray(x, dtype=np.float32)

    orig_predictions = os.path.join(data_path, 'predictions', 'predictions_audio', 'orig', 'orig_' + dataset_type + '.npy')
    mut_predictions_path = os.path.join(data_path, 'predictions', 'predictions_audio', 'mutated')

    if not os.path.exists(orig_predictions):
        raise FileExistsError('Predictions were not found:'+ orig_predictions)
    else:
        ori_predict = np.load(orig_predictions)

    mut_predictions = glob.glob(mut_predictions_path + '/*' + dataset_type + '.npy')

    correct_index = np.where(ori_predict == y)[0]

    yy = np.asarray(y)

    num_mutants = len(mut_predictions)
    num_inputs = len(correct_index)

    count_list = [0 for i in range(num_mutants)]

    i = 0
    for mpred in mut_predictions:
        if not os.path.exists(mpred):
            raise FileExistsError('Predictions were not found:' + mpred)
        else:
            result = np.load(mpred)

        killing_inputs = np.where(yy[correct_index] != result[correct_index])[0]

        num_killing_inputs = len(killing_inputs)

        count_list[i] += num_killing_inputs

        i+=1


    count_list = np.asarray(count_list)

    mut_score_per_mutant = count_list

    mut_score_per_mutant = mut_score_per_mutant / num_inputs

    mutation_score = sum(mut_score_per_mutant) / num_mutants

    return mutation_score


def calc_ms_unityeyes(y, dataset_type):

    orig_predictions = os.path.join(data_path, 'predictions', 'predictions_lenet', 'orig', 'orig_' + dataset_type + '.npy')
    mut_predictions_path = os.path.join(data_path, 'predictions', 'predictions_lenet', 'mutated')

    if not os.path.exists(orig_predictions):
        raise FileExistsError('Predictions were not found:'+ orig_predictions)
    else:
        ori_predict = np.load(orig_predictions)

    mut_predictions = glob.glob(mut_predictions_path + '/*' + dataset_type + '.npy')

    correct_index = np.where(np.degrees(angle_loss_fn_batch(y, ori_predict)) < 5)[0]

    num_inputs = len(correct_index)
    num_mutants = len(mut_predictions)
    count_list = [0 for i in range(num_mutants)]

    i = 0
    for mpred in mut_predictions:
        if not os.path.exists(mpred):
            raise FileExistsError('Predictions were not found:' + mpred)
        else:
            result = np.load(mpred)

        killing_inputs = np.where(np.degrees(angle_loss_fn_batch(y[correct_index], result[correct_index])) >= 5)[0]

        num_killing_inputs = len(killing_inputs)

        count_list[i] += num_killing_inputs

        i+=1


    count_list = np.asarray(count_list)

    mut_score_per_mutant = count_list

    mut_score_per_mutant = mut_score_per_mutant / num_inputs

    mutation_score = sum(mut_score_per_mutant) / num_mutants

    return mutation_score


def calc_ms_mnist(y, dataset_type):
    orig_predictions = os.path.join(data_path, 'predictions', 'predictions_mnist', 'orig', 'orig_' + dataset_type + '.npy')
    mut_predictions_path = os.path.join(data_path, 'predictions', 'predictions_mnist', 'mutated')

    if not os.path.exists(orig_predictions):
        raise FileExistsError('Predictions were not found:' + orig_predictions)
    else:
        ori_predict = np.load(orig_predictions)

    mut_predictions = glob.glob(mut_predictions_path + '/*' + dataset_type + '.npy')

    correct_index = np.where(ori_predict == y)[0]

    num_mutants = len(mut_predictions)
    num_inputs = len(correct_index)

    count_list = [0 for i in range(num_mutants)]

    i = 0
    for mpred in mut_predictions:
        if not os.path.exists(mpred):
            raise FileExistsError('Predictions were not found:' + mpred)
        else:
            result = np.load(mpred)

        killing_inputs = np.where(y[correct_index] != result[correct_index])[0]

        num_killing_inputs = len(killing_inputs)

        count_list[i] += num_killing_inputs

        i+=1

    count_list = np.asarray(count_list)

    mut_score_per_mutant = count_list

    mut_score_per_mutant = mut_score_per_mutant / num_inputs

    mutation_score = sum(mut_score_per_mutant) / num_mutants

    return mutation_score


def calc_ms_udacity(y, dataset_type):
    orig_predictions = os.path.join(data_path, 'predictions', 'predictions_udacity', 'orig', 'orig_' + dataset_type + '.npy')
    mut_predictions_path = os.path.join(data_path, 'predictions', 'predictions_udacity', 'mutated')

    if not os.path.exists(orig_predictions):
        raise FileExistsError('Predictions were not found:'+ orig_predictions)
    else:
        ori_predict = np.load(orig_predictions)

    mut_predictions = glob.glob(mut_predictions_path + '/*' + dataset_type + '.npy')

    correct_index = np.where(abs(y - ori_predict) < 0.3)[0]

    num_mutants = len(mut_predictions)
    num_inputs = len(correct_index)

    count_list = [0 for i in range(num_mutants)]

    i = 0
    for mpred in mut_predictions:
        if not os.path.exists(mpred):
            raise FileExistsError('Predictions were not found:' + mpred)
        else:
            result = np.load(mpred)

        killing_inputs = np.where(abs(y[correct_index] - result[correct_index]) >= 0.3)[0]

        num_killing_inputs = len(killing_inputs)

        count_list[i] += num_killing_inputs

        i += 1


    count_list = np.asarray(count_list)

    mut_score_per_mutant = count_list

    mut_score_per_mutant = mut_score_per_mutant / num_inputs

    mutation_score = sum(mut_score_per_mutant) / num_mutants

    return mutation_score


def get_deepmutationpp_res():
    # Calculate MSs for Audio:
    ts_audio_strong = audio_ts('T')
    ms_audio_strong = calc_ms_audio(ts_audio_strong, "test")
    ms_audio_strong = round(ms_audio_strong, 4)

    ts_audio_weak = audio_ts('W')
    ms_audio_weak = calc_ms_audio(ts_audio_weak, "weak")
    ms_audio_weak = round(ms_audio_weak, 4)

    sens_audio = ((ms_audio_strong - ms_audio_weak) / ms_audio_strong) * 100
    sens_audio = round(sens_audio, 2)
    # print("Audio done")

    # Caclulate MSs for UnityEyes
    ts_ue_strong = unityeyes_ts('T')
    ms_ue_strong = calc_ms_unityeyes(ts_ue_strong, "test")
    ms_ue_strong = round(ms_ue_strong, 4)

    ts_ue_weak = unityeyes_ts('W')
    ms_ue_weak = calc_ms_unityeyes(ts_ue_weak, "weak")
    ms_ue_weak = round(ms_ue_weak, 4)

    sens_ue = ((ms_ue_strong - ms_ue_weak) / ms_ue_strong) * 100
    sens_ue = round(sens_ue, 2)
    # print("Unity done")

    # Caclulate MSs for MNIST
    ts_mnist_strong = mnist_ts('T')
    ms_mnist_strong = calc_ms_mnist(ts_mnist_strong, "test")
    ms_mnist_strong = round(ms_mnist_strong, 4)

    ts_mnist_weak = mnist_ts('W')
    ms_mnist_weak = calc_ms_mnist(ts_mnist_weak, "weak")
    ms_mnist_weak = round(ms_mnist_weak, 4)

    sens_mnist = ((ms_mnist_strong - ms_mnist_weak) / ms_mnist_strong) * 100
    sens_mnist = round(sens_mnist, 2)
    # print("MNIST done")

    # Caclulate MSs for Udacity
    ts_udacity_strong = udacity_ts('T')
    ms_udacity_strong = calc_ms_udacity(ts_udacity_strong, "test")
    ms_udacity_strong = round(ms_udacity_strong, 4)

    ts_udacity_weak = udacity_ts('W')
    ms_udacity_weak = calc_ms_udacity(ts_udacity_weak, "weak")
    ms_udacity_weak = round(ms_udacity_weak, 4)

    sens_udacity = ((ms_udacity_strong - ms_udacity_weak) / ms_udacity_strong) * 100
    sens_udacity = round(sens_udacity, 2)
    # print("Udacity done")

    final_table = pd.DataFrame(columns=["Subject", "DM_Weak_TS", "DM_Strong_TS", "DM_Sensitivity"])
    final_table.loc[0] = ["MN", str(ms_mnist_weak), str(ms_mnist_strong), str(sens_mnist)]
    final_table.loc[1] = ["SR", str(ms_audio_weak), str(ms_audio_strong), str(sens_audio)]
    final_table.loc[2] = ["UD", str(ms_udacity_weak), str(ms_udacity_strong), str(sens_udacity)]
    final_table.loc[3] = ["UE", str(ms_ue_weak), str(ms_ue_strong), str(sens_ue)]

    return final_table



if __name__ == '__main__':
    # Calculate MSs for Audio:
    ts_audio_strong = audio_ts('T')
    ms_audio_strong = calc_ms_audio(ts_audio_strong, "test")
    ms_audio_strong = round(ms_audio_strong, 4)

    ts_audio_weak = audio_ts('W')
    ms_audio_weak = calc_ms_audio(ts_audio_weak, "weak")
    ms_audio_weak = round(ms_audio_weak, 4)

    sens_audio = ((ms_audio_strong-ms_audio_weak)/ms_audio_strong) * 100
    sens_audio = round(sens_audio, 2)
    print("Audio done")


    # Caclulate MSs for UnityEyes
    ts_ue_strong = unityeyes_ts('T')
    ms_ue_strong = calc_ms_unityeyes(ts_ue_strong, "test")
    ms_ue_strong = round(ms_ue_strong, 4)

    ts_ue_weak = unityeyes_ts('W')
    ms_ue_weak = calc_ms_unityeyes(ts_ue_weak, "weak")
    ms_ue_weak = round(ms_ue_weak, 4)

    sens_ue = ((ms_ue_strong-ms_ue_weak)/ms_ue_strong) * 100
    sens_ue = round(sens_ue , 2)
    print("Unity done")


    # Caclulate MSs for MNIST
    ts_mnist_strong = mnist_ts('T')
    ms_mnist_strong = calc_ms_mnist(ts_mnist_strong, "test")
    ms_mnist_strong = round(ms_mnist_strong, 4)

    ts_mnist_weak = mnist_ts('W')
    ms_mnist_weak = calc_ms_mnist(ts_mnist_weak, "weak")
    ms_mnist_weak = round(ms_mnist_weak, 4)

    sens_mnist = ((ms_mnist_strong-ms_mnist_weak)/ms_mnist_strong) * 100
    sens_mnist = round(sens_mnist, 2)
    print("MNIST done")


    # Caclulate MSs for Udacity
    ts_udacity_strong = udacity_ts('T')
    ms_udacity_strong = calc_ms_udacity(ts_udacity_strong, "test")
    ms_udacity_strong = round(ms_udacity_strong, 4)

    ts_udacity_weak = udacity_ts('W')
    ms_udacity_weak = calc_ms_udacity(ts_udacity_weak, "weak")
    ms_udacity_weak = round(ms_udacity_weak, 4)

    sens_udacity = ((ms_udacity_strong-ms_udacity_weak)/ms_udacity_strong) * 100
    sens_udacity = round(sens_udacity, 2)
    print("Udacity done")


    output_file = os.path.join('..', '..', 'Results', 'deepmutationpp_results.csv')

    with open(output_file, 'a') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow(["Subject", "Weak", "Strong", "Sensitivity"])
        # MNIST
        writer.writerow(["MN", str(ms_mnist_weak), str(ms_mnist_strong), str(sens_mnist)])
        # Audio
        writer.writerow(["SR", str(ms_audio_weak), str(ms_audio_strong), str(sens_audio)])
        # Udacity
        writer.writerow(["UD", str(ms_udacity_weak), str(ms_udacity_strong), str(sens_udacity)])
        # UnityEyes
        writer.writerow(["UE", str(ms_ue_weak), str(ms_ue_strong), str(sens_ue)])
