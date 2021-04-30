import os
import datetime
print(datetime.datetime.now())

import subprocess


model_path = os.path.join('..', '..', 'original_models', 'mnist_original_0.h5')
mutants_path = os.path.join('..', '..', 'mutated_models_mnist')

operators = [0, 1, 2, 3, 4, 5, 6, 7]
# operators = [3]
ratio_vals = [0.01, 0.03, 0.05]
ratio = 0.05
num = 400

for operator in operators:
    subprocess.run(["python", "generator.py",
                    "-model_path", model_path,
                    "-data_type", "mnist",
                    "-operator", str(operator),
                    "-ratio", str(ratio),
                    "-save_path", mutants_path,
                    "-num", str(num)])

print(datetime.datetime.now())