from replication_scripts.Model import *
from replication_scripts.utils import get_list_of_files_by_name
import shutil
import glob


def get_prefix_list():
    prefix_set = set()
    for file in glob.glob(model_dir + "*"):
        m_prefix = file[0:file.rindex('_')].replace(model_dir, '')
        prefix_set.add(m_prefix)

    return prefix_set


def get_prediction_array(name_prefix, model_dir):
    files = get_list_of_files_by_name(name_prefix, model_dir)
    prediction_info = []

    for i in range(0, model_num):
        file = os.path.join(model_dir, name_prefix + "_" + str(i) + ".h5")
        prediction_info.append(model.get_prediction_info(file))

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
    np.save(os.path.join(output_dir, mutation_prefix + '_ki.npy'), killing_info)

    return killing_info


def get_model_by_subject():
    if subject_name == 'mnist':
        model = MnistModel()
    elif subject_name == 'movie_recomm':
        model = MovieModel()
    elif subject_name == 'lenet':
        model = UnityModel()
    elif subject_name == 'udacity':
        model = UdacityModel()
    elif subject_name == 'audio':
         model = AudioModel()

    return model


if __name__ == "__main__":
    replication_dir = os.path.join('..', '..')
    for subject_name in ("mnist", "audio", "movie_recomm", "lenet", "udacity"):
        model = get_model_by_subject()
        model_num = 20
        model_dir = os.path.join(replication_dir, 'Models', subject_name)
        list_dir = os.path.join(replication_dir, "Data", "inputs_killability", subject_name)

        output_dir = os.path.join(replication_dir, "Results", "killability_analysis")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.mkdir(output_dir)

        mutation_prefix_list = get_prefix_list()

        original_info_file = os.path.join(output_dir, 'original_prediction_info' + '.npy')
        original_prediction_info = get_prediction_array(subject_name + '_original', model_dir)
        np.save(original_info_file, original_prediction_info)

        for prefix in mutation_prefix_list:
            print("Predicting for Mutation")
            for mutation_prefix in mutation_prefix_list:
                print('mutation_prefix', mutation_prefix)
                if not (os.path.exists(os.path.join(output_dir, mutation_prefix + '_ki.npy'))):
                    mutation_info_file = os.path.join(output_dir, mutation_prefix + '.npy')
                    mutation_prediction_info = get_prediction_array(mutation_prefix, model_dir)
                    np.save(mutation_info_file, mutation_prediction_info)
                    killing_info = get_killing_info(original_prediction_info, mutation_prediction_info, mutation_prefix)
