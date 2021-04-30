import glob
import numpy as np

if __name__ == "__main__":
    models = glob.glob("/Volumes/Samsung_T5/DeepCrime_Models/MNIST/*")
    renamed_models = set()
    renamed_files = set()
    for model in models:
        nm = model.replace("/Volumes/Samsung_T5/DeepCrime_Models/MNIST/", "")
        nm = nm[0:nm.rindex("_")]
        renamed_models.add(nm)

    for file in glob.glob("/Volumes/Samsung_T5/DeepCrime_Redundancy/mnist/*"):
        name = file.replace("/Volumes/Samsung_T5/DeepCrime_Redundancy/mnist/", "").replace(".pickle", "")
        renamed_files.add(name.replace('input_dict_', ''))

    print('renamed_files', renamed_files)
    print('renamed_models', renamed_models)

    print(len(renamed_files))
    print(len(renamed_models))

    for f in renamed_files:
        if f not in renamed_models:
            print(f)