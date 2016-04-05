import pandas as pd
import numpy as np
import os


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


def label_examples(examples, labels=None):
    """
    Returns tuple of examples (X, Y) as numpy arrays:
     X - numpy array n_examples * n_features
     Y - numpy array n_examples * n_labels
     only first 50 samples of each label is considered for setup exercise

    Parameters
    ----------
    examples : pandas data frame represents all examples
    labels : list of all labels
    """
    n_classes = len(labels)
    n_features = 3
    X = np.zeros((0, n_features))
    Y = np.zeros((0, n_classes))

    not_exercises = examples[examples['label'].isnull()]
    xs_not = get_features(not_exercises)
    X = np.append(X, xs_not, axis=0)
    Y = np.append(Y, np.zeros((xs_not.shape[0], n_classes)), axis=0)

    exercises = examples[examples['label'].notnull()]
    if exercises.empty:
        return X, Y

    if labels is None:
        labels_counts = exercises['label'].value_counts()
        labels = [l for l in labels_counts.keys()]

    for i, l in enumerate(labels):
        xs = get_features(exercises[exercises['label'] == l])
        X = np.append(X, xs, axis=0)
        ys = np.zeros((xs.shape[0], n_classes))
        ys[:51, i] = 1
        Y = np.append(Y, ys, axis=0)
    return X, Y


def csv_file_iterator(root_directory):
    """Returns a generator (iterator) of absolute file paths for CSV files in a given directory"""
    for root_path, _, files in os.walk(root_directory, followlinks=True):
        for f in files:
            if f.endswith("csv"):
                yield os.path.join(root_path, f)
