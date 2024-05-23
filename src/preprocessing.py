import numpy as np
import pandas as pd

LABEL_COLUMN = 0
NUM_CLASSES = 10

input_shape = {
    'depth': 1,
    'height': 28,
    'width': 28
}


def normalize(x):
    return x.astype('float32') / 255


def to_one_hot_encoding(y):
    one_hot = np.zeros((len(y), NUM_CLASSES))
    for i, label in enumerate(y):
        one_hot[i, label] = 1
    return one_hot


def one_hotes_to_digits(y):
    digits = []
    for vector in y:
        digits.append(np.argmax(vector))
    return digits


def preprocess_data(x, y, limit=100):
    all_indices = np.array([], dtype=int)
    for label in range(NUM_CLASSES):
        cur_label_indices = np.where(y == label)[0][:limit]
        all_indices = np.concatenate((all_indices, cur_label_indices))
    all_indices = np.array(all_indices, dtype=int)
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), input_shape['depth'], input_shape['height'], input_shape['width'])
    x = normalize(x)
    y = to_one_hot_encoding(y)
    y = y.reshape(len(y), 10, 1)
    return x, y


def create_datasets(df: pd.DataFrame):
    x, y = [df.iloc[:, 1:].values, df.iloc[:, 0].values.flatten()]
    x, y = preprocess_data(x, y)
    return x, y
