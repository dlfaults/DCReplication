import os
import subprocess

model_path = os.path.join('..', '..', 'original_models', 'udacity_original_0.h5')
mutants_path = os.path.join('..', '..', 'mutated_models_udacity')

operators = [0, 1, 2, 3, 4, 5, 6, 7] # not applicable
# operators = [2]
# ratio_vals = [0.01, 0.03, 0.05]
ratio = 0.05
num = 400

for operator in operators:
    subprocess.run(["python", "generator.py",
                    "-model_path", model_path,
                    "-data_type", "lenet",
                    "-operator", str(operator),
                    "-ratio", str(ratio),
                    "-save_path", mutants_path,
                    "-num", str(num)])
