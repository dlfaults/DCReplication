import keras
import h5py
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras import backend as K


def input_reshape_test(x_test, y_test, num_classes):
    img_rows, img_cols = 28, 28
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_test, y_test


def get_mnist_weak_ts(model_dir, dataset_dir, weak_dataset_dir):
    model_file = model_dir
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    model = tf.keras.models.load_model(model_file)
    xx_test, yy_test = input_reshape_test(x_test, y_test, 10)

    predicted = np.asarray(model.predict(xx_test))
    predicted_classes = model.predict_classes(xx_test)

    easy_y = list()
    easy_x = list()

    easy_num = 0
    for i in range (0, len(y_test)):
        confidence = np.max(predicted[i])
        if confidence == 1.0:
            easy_y.append(i)
            easy_x.append(x_test[i])
            easy_num = easy_num + 1

    easy_labels = y_test[easy_y]
    hf = h5py.File(os.path.join(weak_dataset_dir, 'mnist_weak.h5'), 'w')
    hf.create_dataset('x_test', data=easy_x)
    hf.create_dataset('y_test', data=easy_labels)
    hf.close()


if __name__ == '__main__':
    get_mnist_weak_ts(model_dir = os.path.join(), #'/Users/nhumbatova/Documents/UNITY/original_models/'
                           dataset_dir = '',
                           weak_dataset_dir = os.path.join()) #'/Users/nhumbatova/Documents/UNITY/Dataset/weak/'

