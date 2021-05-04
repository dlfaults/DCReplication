import shutil
import pickle
import numpy as np
import os
from replication_scripts.utils import *
from replication_scripts.redundancy_analysis import get_killable_confs


def generate_unstable_inputs_map(dict_dir, array_dir):
    input_dict = dict()
    input_dict_file = os.path.join(dict_dir, subject_name + "_input_dict.pickle")

    for prefix1 in mutation_prefix_list:
        input_dict[prefix1] = {}
        killing_info1 = np.load(os.path.join(array_dir, str(prefix1) + '_ki.npy'), allow_pickle=True)
        killing_probabilities1 = np.sum(killing_info1, axis=0) / len(killing_info1)

        for prefix2 in mutation_prefix_list:
            overlap_num = 0
            indices_to_delete = []
            if prefix1 != prefix2:
                killing_info2 = np.load(os.path.join(array_dir, str(prefix2) + '_ki.npy'))
                killing_probabilities2 = np.sum(killing_info2, axis=0) / len(killing_info2)

                for i in range(0, len(killing_probabilities1)):
                    num_killed1 = int(killing_probabilities1[i] * 20)
                    num_killed2 = int(killing_probabilities2[i] * 20)

                    interval1 = confidence_intervals[num_killed1]
                    interval2 = confidence_intervals[num_killed2]

                    if interval1[0] <= interval2[0] <= interval1[1] or interval2[0] <= interval1[0] <= interval2[1]:
                        indices_to_delete.append(i)
                        overlap_num = overlap_num + 1

                input_dict[prefix1][prefix2] = indices_to_delete

    with open(input_dict_file, 'wb') as f:
        pickle.dump(input_dict, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    replication_dir = os.path.join('..', '..')

    output_dir = os.path.join(replication_dir, "Results", "input_dicts")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.mkdir(output_dir)
    subjects = ["mnist", "audio", "movie_recomm", "udacity", "lenet"]
    confidence_intervals = get_confidence_intervals()

    for subject_name in subjects:
        print("Generating input dict for", subject_name)
        kill_dir = os.path.join(replication_dir, "Data", "inputs_killability", subject_name)
        stats_dir = os.path.join(replication_dir, "Data", "deepcrime_output", subject_name)
        mutation_prefix_list = get_killable_confs(stats_dir, subject_name)
        generate_unstable_inputs_map(output_dir, kill_dir)



