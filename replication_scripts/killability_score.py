import csv
from replication_scripts.constants import subject_params
from replication_scripts.utils import get_outcome_row_index


def get_binary_operator_kill_score(filename, subject, killed_conf):
    if 'delete_td' in filename:
        if killed_conf == 99:
            return 0.01
        return (99 - killed_conf) / 99

    if 'change_learning_rate' in filename:
        lower_lr = subject_params[subject]['lower_lr']
        upper_lr = subject_params[subject]['upper_lr']
        return (killed_conf - lower_lr) / (upper_lr - lower_lr)

    if 'change_epochs' in filename:
        return (killed_conf - 1) / (subject_params[subject]['epochs'] - 1)

    if 'patience' in filename:
        return (killed_conf - 1) / (subject_params[subject]['patience'] - 1)

    if killed_conf == 100:
        return 0.01

    return (100 - killed_conf) / 100


def get_binary_kill_score(filename, subject):
    killed_conf = -1

    row_num = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            killed_conf = row[0]
            row_num = row_num + 1

    if row_num == 1:
        return 0

    return get_binary_operator_kill_score(filename, subject, float(killed_conf))


def get_exh_kill_score(filename):
    index = get_outcome_row_index(filename)

    killed_count = 0
    row_count = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[index] in ('TRUE', 'True'):
                killed_count = killed_count + 1
            row_count = row_count + 1

    ratio = round(killed_count / row_count, 2)
    return ratio


