import glob
import csv
import numpy as np
import collections
from replication_scripts.stats import power, cohen_d
from replication_scripts.constants import subject_params
import pandas as pd
import os
import shutil
from replication_scripts.constants import subject_short_name
from replication_scripts.calculate_deepmutationpp_results import get_deepmutationpp_res


def get_overall_mutation_score(stats_dir, train_stats_dir, name):
    mutation_score = 0
    operator_num = 0
    excluded_num = 0
    score_dict = {}

    csv_file_path = os.path.join(mutation_score_dir, subject_name + '_' + name + '.csv')
    with open(csv_file_path, 'a') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow(['operator_name', 'operator_ms', 'operator_instability_score'])

    for filename in glob.glob(os.path.join(stats_dir, "*")):
        if '.csv' in filename:
            if is_binary_search_operator(filename):
                test_score, ins_score = get_binary_search_operator_mutation_score(filename)
            else:
                test_score, ins_score = get_exhaustive_operator_mutation_score(filename, train_stats_dir)

            if test_score != -1:
                operator = filename.replace(stats_dir + os.path.sep, '')
                score_dict[operator] = {'test_score': test_score,
                                        'ins_score': ins_score}

                if ins_score == 0:
                    mutation_score = mutation_score + test_score
                    operator_num = operator_num + 1

                with open(csv_file_path, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([operator, test_score, ins_score])

    with open(csv_file_path, 'a') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow(['', mutation_score / operator_num, ''])

    return score_dict


def get_binary_search_operator_mutation_score(filename):
    file_short_name = get_file_short_name(filename)
    train_killed_conf = get_killed_conf(os.path.join(train_stats_dir, file_short_name))
    test_killed_conf = get_killed_conf(filename)

    if train_killed_conf == -1:
        # mutant not killed by train set
        return -1, 10

    if test_killed_conf == -1:
        test_killed_conf = get_upper_bound(file_short_name)

    upper_bound = get_upper_bound(file_short_name)

    if train_killed_conf == test_killed_conf:
        mutation_score = 1
    elif upper_bound == train_killed_conf:
        mutation_score = -1
    else:
        mutation_score = round((upper_bound - test_killed_conf) / (upper_bound - train_killed_conf), 2)

    test_power_dict, ins_score_min, ins_score_max = get_power_dict_binary(accuracy_dir, filename, train_killed_conf,
                                                                          test_killed_conf, upper_bound)

    if mutation_score > 1:
        mutation_score = 1

    return mutation_score, abs(ins_score_max) + abs(ins_score_min)


def get_file_short_name(filename):
    return filename[filename.rindex(os.path.sep) + 1:len(filename)]


def get_upper_bound(file_short_name):
    if 'delete_td' in file_short_name:
        return 99

    if 'change_learning_rate' in file_short_name:
        return lower_lr

    if 'change_epochs' in file_short_name or 'change_patience' in file_short_name:
        return 1

    return 100


def get_power_dict_binary(accuracy_dir, stats_file_name, train_killed_conf, test_killed_conf, upper_bound):
    original_file = os.path.join(accuracy_dir, prefix + '.csv')
    original_accuracy = get_accuracy_array_from_file(original_file, 2)
    name = get_replacement_name(stats_file_name, stats_dir, prefix)
    overall_num = 0
    unstable_num = 0
    dict_for_binary = {}
    for filename in glob.glob(os.path.join(accuracy_dir, "*")):
        if name in filename:
            mutation_accuracy = get_accuracy_array_from_file(filename, 2)
            pow = power(original_accuracy, mutation_accuracy)

            mutation_parameter = filename.replace(accuracy_dir, '').replace('.csv', '').replace(name + '_', '').replace(
                'False_', '').replace('_0', '').replace('_3', '').replace('_9', '').replace('_1', '').replace(
                os.path.sep, '')

            if pow >= 0.8:
                dict_for_binary[float(mutation_parameter)] = 's'
            else:
                dict_for_binary[float(mutation_parameter)] = 'uns'

            dict_for_binary = collections.OrderedDict(sorted(dict_for_binary.items()))

    ins_score_min, ins_score_max = get_ins_score(stats_file_name, dict_for_binary, train_killed_conf, test_killed_conf,
                                                 upper_bound)
    return dict_for_binary, ins_score_min, ins_score_max


def get_power_dict_exh(accuracy_dir, stats_file_name):
    original_file = os.path.join(accuracy_dir, prefix + '.csv')
    original_accuracy = get_accuracy_array_from_file(original_file, 2)
    name = get_replacement_name(stats_file_name, stats_dir, prefix)

    dict_for_exh = {}
    for filename in glob.glob(os.path.join(accuracy_dir, "*")):
        if name in filename:
            mutation_accuracy = get_accuracy_array_from_file(filename, 2)
            pow = power(original_accuracy, mutation_accuracy)

            mutation_parameter = filename.replace(accuracy_dir, '').replace('.csv', '').replace(name + '_', '').replace(
                'False_', '').replace(os.path.sep, '')
            if pow >= 0.8:
                dict_for_exh[mutation_parameter] = 's'
            else:
                dict_for_exh[mutation_parameter] = 'uns'

    return dict_for_exh


def get_ins_score(stats_file_name, dict_for_binary, train_killed_conf, test_killed_conf, upper_bound):
    found_first_stable = False
    unstable = 0
    stable = 200
    for key in dict_for_binary:
        if dict_for_binary[key] == 'uns':
            unstable = float(key)
        elif dict_for_binary[key] == 's' and not found_first_stable and float(key) >= test_killed_conf:
            stable = float(key)
            found_first_stable = True

    if stable == 200 or (unstable > stable and not (
            'change_epochs' in stats_file_name or 'change_learning_rate' in stats_file_name or 'change_patience' in stats_file_name)):
        return 1, 1

    if stable < unstable and train_killed_conf < unstable and (
            'change_epochs' in stats_file_name or 'change_learning_rate' in stats_file_name or 'change_patience' in stats_file_name):
        return 0, 0

    if unstable < train_killed_conf and not (
            'change_epochs' in stats_file_name or 'change_learning_rate' in stats_file_name or 'change_patience' in stats_file_name):
        return 0, 0

    if upper_bound - train_killed_conf == 0 or unstable == 0:
        return 0, 0

    if unstable == upper_bound:
        return 1, 1

    if 'change_epochs' in stats_file_name or 'change_patience' in stats_file_name:
        upper_bound = 1

    ins_score_min = round(abs(unstable - train_killed_conf) / abs(upper_bound - train_killed_conf), 2)
    ins_score_max = round(abs(stable - train_killed_conf) / abs(upper_bound - train_killed_conf), 2)
    return ins_score_min, ins_score_max


def get_accuracy_array_from_file(filename, row_index):
    accuracy = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if any(x.strip() for x in row):
                accuracy.append(row[row_index])

    return np.asarray(accuracy).astype(np.float32)


def get_killed_conf(filename):
    killed_conf = -1

    row_num = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            killed_conf = row[0]

            row_num = row_num + 1

    if row_num == 1:
        return -1

    if killed_conf != -1 and train_stats_dir in filename:
        file_short_name = filename[filename.rindex(os.path.sep) + 1:len(filename)]
        killed_name_list.append(file_short_name + '_' + killed_conf)

    return float(killed_conf)


def get_exhaustive_operator_mutation_score(filename, train_stats_dir):
    power_dict_exh_train = get_power_dict_exh(train_accuracy_dir, filename)
    power_dict_exh_test = get_power_dict_exh(accuracy_dir, filename)

    file_short_name = filename[filename.rindex(os.path.sep) + 1:len(filename)]
    train_killed_conf = get_killed_from_csv(os.path.join(train_stats_dir, file_short_name))

    if len(train_killed_conf) == 0:
        return -1, -1

    test_killed_conf = get_killed_from_csv(filename)

    for killed_conf in train_killed_conf:
        if power_dict_exh_train.get(killed_conf) == 'uns':
            train_killed_conf.remove(killed_conf)

    killed_conf = np.intersect1d(train_killed_conf, test_killed_conf)

    if len(train_killed_conf) == 0:
        mutation_score = 0
    else:
        mutation_score = round(len(killed_conf) / len(train_killed_conf), 2)

    if not len(killed_conf) == 0:
        ins_score = get_ins_score_exh(killed_conf, power_dict_exh_test)
    else:
        ins_score = 0

    return mutation_score, ins_score


def get_ins_score_exh(killed_conf, power_dict_exh_test):
    ins_num = 0
    for kc in killed_conf:
        if power_dict_exh_test.get(kc) == 'uns':
            ins_num = ins_num + 1

    return round(ins_num / len(killed_conf), 2)


def get_killed_from_csv(filename):
    index = get_outcome_row_index(filename)

    killed_conf = []
    killed_count = 0
    row_count = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[index] == 'TRUE' or row[index] == 'True':
                if train_stats_dir in filename:
                    file_short_name = filename[filename.rindex(os.path.sep) + 1:len(filename)]

                killed_count = killed_count + 1

                if 'l1' in row[0] or 'l2' in row[0] or 'l1_l2' in row[0]:
                    param = row[0][0: len(row[0]) - 2]
                else:
                    param = row[0]

                killed_conf.append(param)
            row_count = row_count + 1

    ratio = round(killed_count / row_count, 2)
    return killed_conf


def get_outcome_row_index(filename):
    if ('disable_batching' in filename) or ('remove_validation_set' in filename):
        return 2
    else:
        return 3


def is_binary_search_operator(filename):
    operator_list = ['change_label', 'delete_td', 'unbalance_td', 'add_noise',
                     'output_classes_overlap', 'change_epochs', 'change_learning_rate', 'change_patience']
    for operator in operator_list:
        if operator in str(filename):
            return True

    return False


def get_replacement_name(stats_file_name, stats_dir, prefix):
    killed_mutation = stats_file_name.replace(stats_dir + os.path.sep, prefix + '_')
    killed_mutation = killed_mutation.replace('_exssearch.csv', '_mutated0_MP')
    if 'change_epochs' in killed_mutation and 'udacity' in stats_dir:
        killed_mutation = killed_mutation.replace('_binarysearch.csv', '_mutated0_MP_50')
    if 'change_learning_rate' in killed_mutation or 'change_epochs' in killed_mutation:
        killed_mutation = killed_mutation.replace('_binarysearch.csv', '_mutated0_MP_False')
    else:
        killed_mutation = killed_mutation.replace('_binarysearch.csv', '_mutated0_MP')

    killed_mutation = killed_mutation.replace('_nosearch.csv', '_mutated0_MP')
    killed_mutation = killed_mutation.replace('unbalance_td', 'unbalance_train_data')
    killed_mutation = killed_mutation.replace('delete_td', 'delete_training_data')
    killed_mutation = killed_mutation.replace('output_classes_overlap', 'make_output_classes_overlap')
    killed_mutation = killed_mutation.replace('change_patience', 'change_earlystopping_patience')
    return killed_mutation


def get_unstable_operators(score_dict):
    for key in score_dict:
        if score_dict[key]['ins_score'] > 0:
            unstable_operators.append(key)


def get_mutation_score(score_dict):
    overall_mutation_score = 0
    operator_num = 0
    for key in score_dict:
        if key not in unstable_operators:
            overall_mutation_score = overall_mutation_score + score_dict[key]['test_score']
            operator_num = operator_num + 1

    return overall_mutation_score / operator_num


def get_mutation_score(score_dict):
    overall_mutation_score = 0
    operator_num = 0
    for key in score_dict:
        if key not in unstable_operators:
            overall_mutation_score = overall_mutation_score + score_dict[key]['test_score']
            operator_num = operator_num + 1

    return overall_mutation_score / operator_num


if __name__ == "__main__":
    replication_dir = os.path.join('..', '..')
    mutation_score_dir = os.path.join(replication_dir, 'Results', 'mutation_score')

    if os.path.exists(mutation_score_dir):
        shutil.rmtree(mutation_score_dir)

    os.mkdir(mutation_score_dir)

    subjects = ["mnist", "audio", "udacity", "lenet"]
    final_table = pd.DataFrame(columns=["Subject", "DC_Weak_TS", "DC_Strong_TS", "DC_Sensitivity"])
    i = 0
    for subject_name in subjects:
        prefix = subject_name
        epochs = subject_params[subject_name]['epochs']
        lower_lr = subject_params[subject_name]['lower_lr']
        upper_lr = subject_params[subject_name]['upper_lr']

        train_accuracy_dir = os.path.join(replication_dir, "Data", "deepcrime_output", subject_name, "results_train")
        train_stats_dir = os.path.join(train_accuracy_dir, 'stats')

        killed_name_list = []
        accuracy_dir = os.path.join(replication_dir, "Data", "deepcrime_output", subject_name, "results_strong_ts")
        stats_dir = os.path.join(accuracy_dir, 'stats')
        score_dict_strong = get_overall_mutation_score(stats_dir, train_stats_dir, "results_strong_ts")

        killed_name_list = []
        accuracy_dir = os.path.join(replication_dir, "Data", "deepcrime_output", subject_name, "results_weak_ts")
        stats_dir = os.path.join(accuracy_dir, 'stats')
        score_dict_weak = get_overall_mutation_score(stats_dir, train_stats_dir, "results_weak_ts")

        unstable_operators = []
        get_unstable_operators(score_dict_strong)
        get_unstable_operators(score_dict_weak)

        instability_txt_file = os.path.join(mutation_score_dir, subject_name + '_unstable_operators.txt')
        text_file = open(instability_txt_file, "w")
        text_file.write("Number of unstable operators: %i \n" % len(set(unstable_operators)))
        if len(set(unstable_operators)) > 0:
            text_file.write(str(set(unstable_operators)))
        text_file.close()

        ms_score_strong = round(get_mutation_score(score_dict_strong), 4)
        ms_score_weak = round(get_mutation_score(score_dict_weak), 4)
        sensitivity = round((ms_score_strong - ms_score_weak) * 100 / ms_score_strong, 2)

        final_table.loc[i] = [subject_short_name[subject_name], ms_score_weak, ms_score_strong, sensitivity]
        i = i + 1

    dm_final_table = get_deepmutationpp_res()
    final_table = final_table.join(dm_final_table.set_index('Subject'), on='Subject')
    final_table.to_csv(os.path.join(mutation_score_dir, "table6.csv"))
