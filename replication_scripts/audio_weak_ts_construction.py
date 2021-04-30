import math
import numpy as np
import h5py
import os
import shutil
from pathlib import Path

import keras
from keras import backend as K
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_noise_sample(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == SAMPLING_RATE:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / SAMPLING_RATE)
        sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
        return sample
    else:
        print("Sampling rate for {} is incorrect. Ignoring it".format(path))
        return None


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        print(audio.shape)
        print(noise.shape)
        print((noise * prop * scale).shape)
        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def get_audio_easy_indexes(model, test_ds):
    index = 0
    #easy_y = []
    overall_num = 0
    element_index = 0
    list_of_confident = []
    for element in test_ds:
        print("element_index:" + str(element_index))
        predicted = np.asarray(model.predict(element))
        overall_num = overall_num + len(predicted)
        correct_prediction_array = element[1].numpy()

        for i in range(0, len(predicted)):
            prediction = np.argmax(predicted[i])
            confidence = np.max(predicted[i])
            correct_prediction = correct_prediction_array[i]
            print(str(prediction) + ' ' + str(correct_prediction) + ' ' + str(confidence))
            if prediction == correct_prediction and confidence == 1.0:
                print("this is the case:" + str(element_index) + ' ' + str(i))
                list_of_confident.append(int(str(element_index) + '000' + str(i)))
                #easy_y.append(element[1][i])

        element_index = element_index + 1
    print('overall num:' + str(overall_num))
    return list_of_confident


def get_audio_eyes_weak_ts(model_dir, dataset_dir, weak_dataset_dir):
    SAMPLING_RATE = 16000
    DATASET_ROOT = dataset_dir

    # The folders in which we will put the audio samples and the noise samples
    AUDIO_SUBFOLDER = "audio"
    NOISE_SUBFOLDER = "noise"

    DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
    DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

    VALID_SPLIT = 0.1

    # Percentage of samples to use for testing
    TEST_SPLIT = 0.2

    SHUFFLE_SEED = 43

    SCALE = 0.5

    BATCH_SIZE = 128
    EPOCHS = 100

    if os.path.exists(DATASET_AUDIO_PATH) is False:
        os.makedirs(DATASET_AUDIO_PATH)

    if os.path.exists(DATASET_NOISE_PATH) is False:
        os.makedirs(DATASET_NOISE_PATH)

    for folder in os.listdir(DATASET_ROOT):
        if os.path.isdir(os.path.join(DATASET_ROOT, folder)):
            if folder in [AUDIO_SUBFOLDER, NOISE_SUBFOLDER]:
                continue
            elif folder in ["other", "_background_noise_"]:
                shutil.move(
                    os.path.join(DATASET_ROOT, folder),
                    os.path.join(DATASET_NOISE_PATH, folder),
                )
            else:
                shutil.move(
                    os.path.join(DATASET_ROOT, folder),
                    os.path.join(DATASET_AUDIO_PATH, folder),
                )

    # Get the list of all noise files
    noise_paths = []
    for subdir in os.listdir(DATASET_NOISE_PATH):
        subdir_path = Path(DATASET_NOISE_PATH) / subdir
        if os.path.isdir(subdir_path):
            noise_paths += [
                os.path.join(subdir_path, filepath)
                for filepath in os.listdir(subdir_path)
                if filepath.endswith(".wav")
            ]

    print(
        "Found {} files belonging to {} directories".format(
            len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))
        )
    )

    command = (
            "for dir in `ls -1 " + DATASET_NOISE_PATH + "`; do "
                                                        "for file in `ls -1 " + DATASET_NOISE_PATH + "/$dir/*.wav`; do "
                                                                                                     "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
                                                                                                     "$file | grep sample_rate | cut -f2 -d=`; "
                                                                                                     "if [ $sample_rate -ne 16000 ]; then "
                                                                                                     "ffmpeg -hide_banner -loglevel panic -y "
                                                                                                     "-i $file -ar 16000 temp.wav; "
                                                                                                     "mv temp.wav $file; "
                                                                                                     "fi; done; done"
    )

    os.system(command)

    noises = []
    for path in noise_paths:
        sample = load_noise_sample(path)
        if sample:
            noises.extend(sample)
    noises = tf.stack(noises)

    print(
        "{} noise files were split into {} noise samples where each is {} sec. long".format(
            len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE
        )
    )

    class_names = os.listdir(DATASET_AUDIO_PATH)
    print("Our class names: {}".format(class_names, ))

    audio_paths = []
    labels = []
    for label, name in enumerate(class_names):
        print("Processing speaker {}".format(name, ))
        dir_path = Path(DATASET_AUDIO_PATH) / name
        speaker_sample_paths = [
            os.path.join(dir_path, filepath)
            for filepath in os.listdir(dir_path)
            if filepath.endswith(".wav")
        ]
        audio_paths += speaker_sample_paths
        labels += [label] * len(speaker_sample_paths)

    print(
        "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
    )

    # Shuffle
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(audio_paths)
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(labels)

    # Split into training and validation
    num_val_samples = int(VALID_SPLIT * len(audio_paths))
    num_test_samples = int(TEST_SPLIT * (len(audio_paths) - num_val_samples))
    print("Using {} files for training.".format(len(audio_paths) - num_val_samples))

    train_audio_paths = audio_paths[:-num_val_samples]
    train_labels = labels[:-num_val_samples]

    print("Using {} files for validation.".format(num_val_samples))
    valid_audio_paths = audio_paths[-num_val_samples:]
    valid_labels = labels[-num_val_samples:]

    test_audio_paths = train_audio_paths[-num_test_samples:]
    test_labels = train_labels[-num_test_samples:]

    train_audio_paths = train_audio_paths[:-num_test_samples]
    train_labels = train_labels[:-num_test_samples]

    train_audio_paths: 'x_train'
    train_labels: 'y_train'

    # Create 2 datasets, one for training and the other for validation
    train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
    train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)

    valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

    test_ds_orig = paths_and_labels_to_dataset(test_audio_paths, test_labels)
    test_ds = test_ds_orig.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

    # Add noise to the training set
    train_ds = train_ds.map(
        lambda x, y: (add_noise(x, noises, scale=SCALE), y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Transform audio wave to the frequency domain using `audio_to_fft`
    train_ds = train_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    valid_ds = valid_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = test_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    print("test_ds")
    print(test_ds)

    list_of_lists = []
    for i in range(0, 3):
        model_file = os.path.join(model_dir, 'audio_original_' + str(i) + '.h5')
        model = tf.keras.models.load_model(model_file)
        scores = model.evaluate(test_ds, verbose=0)
        print(scores)
        list_of_confident = get_audio_easy_indexes(model, test_ds)
        print(len(list_of_confident))
        list_of_lists.append(list_of_confident)

    intersection = set(list_of_lists[0])
    for index in range(0, len(list_of_lists)):
        if index < len(list_of_lists) - 1:
            intersection = set(intersection & set(list_of_lists[index + 1]))
            print(intersection)
        index = index + 1

    print(len(intersection))

    easy_num = 0

    easy_x = np.empty([686, 8000, 1], dtype='float32')


    print("easy_num:" + str(easy_num))
    print("easy_x:" + str(len(easy_x)) + " " + str(easy_x.shape))
    print("easy_y:" + str(len(easy_y)))

    easy_y = np.asarray(easy_y).reshape(len(easy_y),)

    np.save(os.path.join(weak_dataset_dir, 'audio_easy_x.npy'), easy_x)
    np.save(os.path.join(weak_dataset_dir, 'audio_easy_y.npy'), easy_y)


if __name__ == '__main__':
    get_audio_eyes_weak_ts(model_dir = os.path.join(), #'/home/ubuntu/mutation-tool/trained_models/'
                            dataset_dir = os.path.join(), #"/home/ubuntu/16000_pcm_speeches"
                            weak_dataset_dir= os.path.join()) #""