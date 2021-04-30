import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
from tensorflow.keras.models import load_model
from tensorflow import math

import numpy as np
from sklearn.model_selection import train_test_split
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_label_buckets(y_train):
    std_array = np.std(y_train, axis=0)

    unique_label_list, unique_inverse, unique_counts = np.unique(y_train, return_counts=True,
                                                                 return_inverse=True,
                                                                 axis=0)
    index = 0
    bucket1 = []
    bucket2 = []
    bucket3 = []
    bucket4 = []
    bucket5 = []
    bucket6 = []
    bucket7 = []
    bucket8 = []
    bucket9 = []

    print("Std label 0:" + str(np.degrees(std_array[0])))
    print("Std label 1:" + str(np.degrees(std_array[1])))

    for label in unique_label_list:
        args = np.argwhere(unique_inverse == index)
        if label[0] < -std_array[0]:
            if label[1] < -std_array[1]:
                bucket1.extend(args.flatten().tolist())
            elif label[1] >= -std_array[1] and label[1] < std_array[1]:
                bucket2.extend(args.flatten().tolist())
            elif label[1] >= std_array[1]:
                bucket3.extend(args.flatten().tolist())
        elif -std_array[0] <= label[0] < std_array[0]:
            if label[1] < -std_array[1]:
                bucket4.extend(args.flatten().tolist())
            elif -std_array[1] <= label[1] < std_array[1]:
                bucket5.extend(args.flatten().tolist())
            elif label[1] >= std_array[1]:
                bucket6.extend(args.flatten().tolist())
        elif label[0] >= std_array[0]:
            if label[1] < -std_array[1]:
                bucket7.extend(args.flatten().tolist())
            elif -std_array[1] <= label[1] < std_array[1]:
                bucket8.extend(args.flatten().tolist())
            elif label[1] >= std_array[1]:
                bucket9.extend(args.flatten().tolist())
        index = index + 1

    buckets = [bucket1, bucket2, bucket3, bucket4, bucket5, bucket6, bucket7, bucket8, bucket9]

    unique_label_list = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    unique_inverse = [-1] * len(y_train)
    unique_count = [-1] * len(unique_label_list)
    bucket_ind = 0
    for bucket in buckets:
        unique_count[bucket_ind] = len(bucket)
        for element in bucket:
            unique_inverse[element] = bucket_ind

        bucket_ind = bucket_ind + 1
    return unique_label_list, np.asarray(unique_count), np.asarray(unique_inverse), buckets


def angle_loss_fn(y_true, y_pred):
    x_p = math.sin(y_pred[:, 0]) * math.cos(y_pred[:, 1])
    y_p = math.sin(y_pred[:, 0]) * math.sin(y_pred[:, 1])
    z_p = math.cos(y_pred[:, 0])

    x_t = math.sin(y_true[:, 0]) * math.cos(y_true[:, 1])
    y_t = math.sin(y_true[:, 0]) * math.sin(y_true[:, 1])
    z_t = math.cos(y_true[:, 0])

    norm_p = math.sqrt(x_p * x_p + y_p * y_p + z_p * z_p)
    norm_t = math.sqrt(x_t * x_t + y_t * y_t + z_t * z_t)

    dot_pt = x_p * x_t + y_p * y_t + z_p * z_t

    angle_value = dot_pt/(norm_p * norm_t)
    angle_value = tf.clip_by_value(angle_value, -0.99999, 0.99999)

    loss_val = (math.acos(angle_value))

    return loss_val


def get_unity_eyes_weak_ts(model_dir, dataset_dir, weak_dataset_dir):
    x_img = np.load(os.path.join(dataset_dir, 'dataset_x_img.npy'))
    x_head_angles = np.load(os.path.join(dataset_dir, 'dataset_x_head_angles_np.npy'))
    y_gaze_angles = np.load(os.path.join(dataset_dir, 'dataset_y_gaze_angles_np.npy'))

    x_img_train, x_img_test, x_ha_train, x_ha_test, y_gaze_train, y_gaze_test = train_test_split(x_img, x_head_angles,
                                                                                                 y_gaze_angles,
                                                                                                 test_size=0.2,
                                                                                                 random_state=42)

    list_of_losses = []

    for i in range(0, 20):
        model_file = os.path.join(model_dir, 'lenet_original_' + str(i) + '.h5')

        model = load_model(model_file, compile=False)
        x_img_test = x_img_test
        x_ha_test = x_ha_test
        y_gaze_test = y_gaze_test
        predicted = np.asarray(model.predict([x_img_test, x_ha_test]))

        loss = angle_loss_fn(y_gaze_test, predicted)
        list_of_losses.append(loss)


    list_of_losses = np.asarray(list_of_losses)
    std_of_losses = np.std(list_of_losses, axis=0)

    unique_label_list, unique_counts, unique_inverse, buckets = get_label_buckets(y_gaze_test)

    buckets = [buckets[4]]

    buckets_size_weak = 4000

    chosen_elements = []

    for i, bucket in enumerate(buckets):
        bucket_loss = std_of_losses[bucket]

        sorted_by_loss = np.argsort(bucket_loss)

        print(sorted_by_loss)
        print(bucket_loss[sorted_by_loss])

        sorted_by_loss_subset = sorted_by_loss[-buckets_size_weak:]

        bucket = np.asarray(bucket)
        chosen_subset = bucket[sorted_by_loss_subset]

        chosen_elements.extend(chosen_subset)

    x_img_test_sub = x_img_test[chosen_elements]
    x_ha_test_sub = x_ha_test[chosen_elements]
    y_gaze_test_sub = y_gaze_test[chosen_elements]

    # save the first input to the prediction model: eye region images
    np.save(os.path.join(weak_dataset_dir, 'ue_weak_ts_x_img.npy'), x_img_test_sub)
    # save the second input to the prediction model: head angles
    np.save(os.path.join(weak_dataset_dir, 'ue_weak_ts_x_ha.npy'), x_ha_test_sub)
    # save the ground truth: eye rotation angles
    np.save(os.path.join(weak_dataset_dir, 'ue_weak_ts_y_eg.npy'), y_gaze_test_sub)


if __name__ == '__main__':
    get_unity_eyes_weak_ts(model_dir = os.path.join(), #'/Users/nhumbatova/Documents/UNITY/original_models/'
                           dataset_dir = os.path.join(), #'/Users/nhumbatova/Documents/UNITY/Dataset/'
                           weak_dataset_dir = os.path.join()) #'/Users/nhumbatova/Documents/UNITY/Dataset/weak/'
