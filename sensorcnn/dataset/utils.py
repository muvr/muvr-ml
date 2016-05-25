import pandas as pd
import numpy as np
import os
from sklearn.cross_validation import train_test_split


def load_from_csv(filename):
    """Load data from csv file and return pandas data frame"""
    names = ['x', 'y', 'z', 'label', '', '', '']
    dtype = {'x': pd.np.float32, 'y': pd.np.float32, 'z': pd.np.float32}
    examples = pd.read_csv(filename, header=None, names=names, dtype=dtype)
    return examples


def get_features(df, *features):
    """
    Extract features from pandas data frame as a numpy array

    Parameters
    ----------
    df : pandas data frame
    features : list of the labels in the data frame
    """
    if not features:
        features = ['x', 'y', 'z']

    return df[list(features)].as_matrix()


def csv_file_iterator(root_directory):
    """Returns a generator (iterator) of absolute file paths for CSV files in a given directory"""
    for root_path, _, files in os.walk(root_directory, followlinks=True):
        for f in files:
            if f.endswith("csv"):
                yield os.path.join(root_path, f)


def augment_examples(X, Y, new_sample_size=50, label_merge=np.average):
    """"""
    assert X.shape[0] == Y.shape[0], "X and Y lengthes don't match"
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_labels = Y.shape[1]
    end = n_samples - (n_samples % new_sample_size)
    new_X = X[:end, :].reshape((int(end/new_sample_size), int(n_features*new_sample_size)))
    ys = []
    for y in Y[:end, :].reshape((int(end/new_sample_size), int(n_labels*new_sample_size))):
        ys.append(label_merge(y.reshape((new_sample_size, n_labels)), axis=0))

    new_Y = np.array(ys).astype(int)
    return new_X, new_Y


def boost(x, n, partition):
    repeated = np.repeat(x[0:partition, :], n-1, axis=0)
    return np.append(x, repeated, axis=0)


def split(X, Y, test_size=0.3, random_state=42):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test
