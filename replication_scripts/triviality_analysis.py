import numpy as np
import pandas as pd
import csv
import shutil

from replication_scripts.Model import MnistModel
from replication_scripts.utils import *
from replication_scripts.constants import operator_name_dict
from replication_scripts.killability_score import *


def analyse_triviality(model_dir):
    column_names = ["Operator", "Mutation_Name", subject_name +'_TS']
    df = pd.DataFrame(columns=column_names)

    original_info_file = os.path.join(output_dir, 'original_prediction_info.npy')
    if not os.path.exists(original_info_file):
        original_prediction_info = get_prediction_array(subject_name + '_original', model_dir)
        np.save(original_info_file, original_prediction_info)
    else:
        original_prediction_info = np.load(original_info_file)

    for mutation_prefix in mutation_prefix_list:
        operator_name = mutation_prefix[0:mutation_prefix.index('_mutated0')].replace(subject_name + '_', '')
        operator_name = operator_name.replace('earlystopping_patience', 'patience')
        path = os.path.join(output_dir, mutation_prefix + '_ki.npy')
        if not os.path.exists(os.path.join(output_dir, mutation_prefix + '_ki.npy')):
            mutation_info_file = os.path.join(output_dir, mutation_prefix + '.npy')
            if not (os.path.exists(mutation_info_file)):
                mutation_prediction_info = get_prediction_array(mutation_prefix, model_dir)
                np.save(mutation_info_file, mutation_prediction_info)
            else:
                mutation_prediction_info = np.load(mutation_info_file)
            killing_info = get_killing_info(original_prediction_info, mutation_prediction_info, mutation_prefix)
        else:
            killing_info = np.load(os.path.join(output_dir, mutation_prefix + '_ki.npy'))

        killing_probabilities = np.sum(killing_info, axis=0) / len(killing_info)

        expected_value = np.sum(killing_probabilities)
        triviality_score = expected_value / len(killing_probabilities)
        df = df.append({'Operator': operator_name, 'Mutation_Name': mutation_prefix, subject_name +'_TS': triviality_score}, ignore_index=True)

    triviality_csv_file = os.path.join(triviality_output_dir, subject_name + '_triviality.csv')
    df.to_csv(triviality_csv_file)

    trivial_mutants = df[df[subject_name +'_TS'] >= 0.9]
    trivial_txt_file = os.path.join(triviality_output_dir, subject_name + '_trivial_mutants.txt')
    text_file = open(trivial_txt_file, "w")
    text_file.write("Number of trivial mutants: %i \n" % trivial_mutants.shape[0])
    if trivial_mutants.shape[0] > 0:
        text_file.write(str(trivial_mutants['Mutation_Name'].tolist()))
    text_file.close()

    df.set_index('Operator')
    ats = df.groupby('Operator', as_index=False).mean()

    return ats


def get_prediction_array(name_prefix, model_dir):
    files = get_list_of_files_by_name(name_prefix, model_dir)
    prediction_info = []

    for i in range(0, model_num):
        file = os.path.join(model_dir, name_prefix + "_" + str(i) + ".h5")
        #prediction_info.append(model.get_prediction_info(file))

    return prediction_info


def get_killing_info(original_prediction_info, mutation_prediction_info, mutation_prefix):
    killing_info = []

    for i in range(0, len(original_prediction_info)):
        killing_array = np.empty(len(original_prediction_info[i]))
        for j in range(0, len(original_prediction_info[i])):
            if (original_prediction_info[i][j] is True) and (mutation_prediction_info[i][j] is False):
                killing_array[j] = 1
            else:
                killing_array[j] = 0
        killing_info.append(killing_array)

    killing_info = np.asarray(killing_info)
    np.save(os.path.join(output_dir, mutation_prefix, '_ki.npy', killing_info))

    return killing_info


def calculate_killability_score(subject_name):
    stat_file_loc = os.path.join(replication_dir, "Data", "deepcrime_output", subject_name ,"results_train", "stats")
    operators = []
    killability_scores = []
    for file in glob.glob(stat_file_loc + "/*"):
        filename = file.replace('_binarysearch.csv', '').replace('_nosearch.csv', '').replace('_exssearch.csv', '')
        filename = filename[filename.rindex(os.path.sep) + 1:len(filename)]
        filename = filename.replace('delete_td', 'delete_training_data')
        filename = filename.replace('unbalance_td', 'unbalance_train_data')
        filename = filename.replace('output_classes_overlap', 'make_output_classes_overlap')

        operators.append(filename)
        if is_binary_search_operator(file):
            ks_score = get_binary_kill_score(file, subject_name)
        else:
            ks_score = get_exh_kill_score(file)
        killability_scores.append(round(ks_score, 2))
    df = pd.DataFrame({'Operator': operators, subject_name + '_KS':killability_scores})
    return df


if __name__ == "__main__":
    #replication_dir = "/Volumes/Samsung_T5/deepcrime_replication/"
    replication_dir = os.path.join('..', '..')
    model_num = 20
    triviality_output_dir = os.path.join(replication_dir, "Results", "triviality_analysis")
    if os.path.exists(triviality_output_dir):
        shutil.rmtree(triviality_output_dir)

    os.mkdir(triviality_output_dir)

    subjects = ["mnist", "audio", "movie_recomm", "udacity", "lenet"]
    final_table = pd.DataFrame({'Operator': list(operator_name_dict.keys())})
    final_table.set_index('Operator')

    for subject_name in subjects:
        print("Performing Triviality Analysis for", subject_name)
        output_dir = os.path.join(replication_dir, "Data", "inputs_killability", subject_name)
        mutation_prefix_list = get_prefix_list(output_dir)
        column_ats = analyse_triviality(output_dir)
        column_ks = calculate_killability_score(subject_name)
        final_table = final_table.join(column_ks.set_index('Operator'), on='Operator')
        final_table = final_table.join(column_ats.set_index('Operator'), on='Operator')
        final_table.loc[final_table[subject_name + '_KS'] == 0, subject_name + '_TS'] = ''

    final_table.to_csv(os.path.join(triviality_output_dir, "table4.csv"))