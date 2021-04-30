import glob
import os
from keras.models import load_model
import keras.backend as K
import gc
import time
import numpy as np
import matplotlib.pyplot as plt


def check_data(ori_model_path, mutants_path, x, y, save_path, dataset):
    model_path = mutants_path
    model_path = glob.glob(model_path + '/*.h5')

    count_list = [0 for i in range(len(model_path))]
    orig_predictions = os.path.join('..', '..', "predictions", "predictions_mnist", "orig", 'orig_' + dataset + '.npy')
    mut_predctions = os.path.join('..', '..', "predictions", "predictions_mnist", 'mutated')

    if not os.path.exists(orig_predictions):
        ori_model = load_model(ori_model_path)

        ori_predict = ori_model.predict(x).argmax(axis=-1)

        np.save(orig_predictions, ori_predict)
    else:
        ori_predict = np.load(orig_predictions)

    correct_index = np.where(ori_predict == y)[0]
    # print("Number of correctly classified by original")
    # print(len(correct_index))

    start_time = time.process_time()

    i = 0
    for path in model_path:
        # print((path))

        model_name = (path.split("\\"))[-1].replace(".h5", "")
        # print(model_name)

        mut_predctions_path = os.path.join(mut_predctions, model_name + "_" + dataset + ".npy")

        if not os.path.exists(mut_predctions_path):
            model = load_model(path)
            result = model.predict(x).argmax(axis=-1)
            np.save(mut_predctions_path, result)

            K.clear_session()
            del model
            gc.collect()
        else:
            result = np.load(mut_predctions_path)

        killing_inputs = np.where(y[correct_index] != result[correct_index])[0]

        num_inputs = len(killing_inputs)

        count_list[i] += num_inputs

        i+=1

    elapsed = (time.process_time() - start_time)
    print("running time: ", elapsed)

    count_list = np.asarray(count_list)

    num_mutants = len(model_path)

    num_inputs = len(correct_index)

    mut_score_per_mutant = count_list

    mut_score_per_mutant = mut_score_per_mutant / num_inputs

    mutation_score = sum(mut_score_per_mutant) / num_mutants

    sem = np.std(mut_score_per_mutant, axis=0) / np.math.sqrt(num_mutants)

    magn = (sem/mutation_score) * 100

    lines = []
    lines.append("Mutation Score: %s \n" % mutation_score)
    lines.append("SEM: %s \n" % sem)
    lines.append("Magnitude: %s \n" % magn)
    with open(os.path.join(save_path, "mut_score_calc_%s_check.txt" % dataset), "w") as text_file:
        text_file.writelines(lines)

    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    a = ax1.boxplot(mut_score_per_mutant)
    plt.savefig(os.path.join(save_path, "mut_score_%s_check.png" % dataset))