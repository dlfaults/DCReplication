import shutil
import pickle
from replication_scripts.utils import *
import networkx as nx


def generate_unstable_inputs_map(dict_dir, array_dir):
    input_dict = dict()
    input_dict_file = os.path.join(dict_dir, "input_dict_new.pickle")

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


def get_replacement_name(stats_file_name, stats_dir, prefix):
    killed_mutation = stats_file_name.replace(stats_dir, prefix + '_')
    killed_mutation = killed_mutation.replace('_exssearch.csv', '_mutated0_MP')
    if subject_name == 'udacity' and 'change_epochs' in killed_mutation:
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


def get_killed_from_csv(mutation_prefix_list, filename, stats_dir):
    if is_binary_search_operator(filename):
        kill_index = 5
        value_index = 2
    else:
        kill_index = get_outcome_row_index(filename)
        value_index = 0

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 2 and (row[kill_index] == 'TRUE' or row[kill_index] == 'True'):
                if len(row[value_index].strip()) == 0:
                    parameter = row[value_index - 1]
                else:
                    parameter = row[value_index]
                    if 'batch_size' in filename:
                        parameter = parameter + '_' + parameter

                short_name = filename[filename.rindex(os.path.sep) + 1:len(filename)]
                file_name = get_replacement_name(short_name, stats_dir, 'lenet')

                final_name = subject_name + '_' + file_name + '_' + parameter
                if 'remove_validation_set' in file_name or 'disable_batching' in file_name:
                    final_name = final_name.replace('_' + parameter, '')

                mutation_prefix_list.add(final_name)


def get_killable_confs(dir):
    mutation_prefix_list = set()
    sub_dirs = ['results_train', 'results_strong_ts', 'results_weak_ts']
    for subdir in sub_dirs:
        for filename in glob.glob(os.path.join(dir, subdir, "stats" + os.path.sep + "*")):
            file_short_name = filename[filename.rindex(os.path.sep) + 1:len(filename)]
            get_killed_from_csv(mutation_prefix_list, filename, dir)

    return mutation_prefix_list


def analyse_redundancy(array_dir, threshold=100):
    rs_csv_file = os.path.join(output_dir, 'redundancy_score' + str(threshold) + '.csv')
    g = nx.DiGraph()

    unstable_inputs_dict = get_dict(array_dir)

    for prefix1 in mutation_prefix_list:
        if prefix1 not in g:
            g.add_node(prefix1)

        killing_info1 = np.load(os.path.join(array_dir,  str(prefix1) + '_ki.npy'))
        killing_probabilities1 = np.sum(killing_info1, axis=0) / len(killing_info1)

        for prefix2 in mutation_prefix_list:
            if prefix1 != prefix2 and prefix1 != prefix2 + '.0':
                killing_info2 = np.load(os.path.join(array_dir, str(prefix2) + '_ki.npy'))
                killing_probabilities2 = np.sum(killing_info2, axis=0) / len(killing_info2)
                indices_to_delete = unstable_inputs_dict[prefix1][prefix2]

                if len(indices_to_delete) < len(killing_probabilities1):
                    killing_probabilities1_copy = np.delete(killing_probabilities1, indices_to_delete)
                    killing_probabilities2_copy = np.delete(killing_probabilities2, indices_to_delete)

                    if len(killing_probabilities1_copy) != 0 and len(killing_probabilities2_copy) != 0:
                        comparison_array = np.less_equal(killing_probabilities1_copy, killing_probabilities2_copy)
                        num_true = len(np.argwhere(comparison_array == True))

                        sc_low, sc_high = smp.proportion_confint(num_true, len(killing_probabilities1_copy), alpha=0.10, method='wilson')
                        error = (sc_high - sc_low) / 2

                        if error < 0.05 and num_true >= (threshold / 100) * len(killing_probabilities1_copy):
                            g.add_edge(prefix1, prefix2)

    nodes = np.sort(g.nodes())

    redundant_nodes = []
    non_redundant_nodes = []
    for node in nodes:
        if g.in_degree(node) == 0:
            non_redundant_nodes.append(str(node))
        else:
            redundant_nodes.append(str(node))

    return redundant_nodes, non_redundant_nodes


def get_dict(array_dir):
    input_dict = dict()
    input_dict_file = os.path.join(dict_dir, "input_dict_new.pickle")
    with open(input_dict_file, 'rb') as f:
        input_dict = pickle.load(f)
    return input_dict


if __name__ == "__main__":
    replication_dir = os.path.join('..', '..')

    redundancy_output_dir = os.path.join(replication_dir, "Results", "redundancy_analysis")
    if os.path.exists(redundancy_output_dir):
        shutil.rmtree(redundancy_output_dir)

    os.mkdir(redundancy_output_dir)
    subjects = ["mnist", "audio", "movie_recomm", "udacity", "lenet"]
    confidence_intervals = get_confidence_intervals()

    final_table = pd.DataFrame(columns=['ID', 'Killable Confs', 'Redundant', 'Non Redundant'])

    i = 0
    for subject_name in subjects:
        print("Performing Redundancy Analysis for", subject_name)
        stats_dir = os.path.join(replication_dir, "Data", "deepcrime_output", subject_name)
        mutation_prefix_list = get_killable_confs(stats_dir)
        output_dir = os.path.join(replication_dir, "Data", "inputs_killability", subject_name)
        dict_dir = os.path.join(replication_dir, "Data", "input_dicts", subject_name)
        #generate_unstable_inputs_map(dict_dir, output_dir)

        redundant_nodes, non_redundant_nodes = analyse_redundancy(output_dir)
        assert (len(mutation_prefix_list) == len(redundant_nodes) + len(non_redundant_nodes))

        write_list_to_csv(os.path.join(redundancy_output_dir, subject_name + '_redundant.csv'), redundant_nodes)
        write_list_to_csv(os.path.join(redundancy_output_dir, subject_name + '_non_redundant.csv'), non_redundant_nodes)

        final_table.loc[i] = [subject_name, len(mutation_prefix_list), len(redundant_nodes), len(non_redundant_nodes)]
        i = i + 1

    final_table.to_csv(os.path.join(redundancy_output_dir, 'table5.csv'))


