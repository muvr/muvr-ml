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


def set_setup_column(df, start=0, end=50):
    df['setup_label'] = 0
    n = 0
    for i, (s, is_excercise) in enumerate(zip(df[['label']].values, df['label'].notnull())):
        if is_excercise:
            n += 1
            if start <= n <= end:
                df.loc[i,'setup_label']= "setup_%s" % s[0]

        else:
            n = 0


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


def map_labels(label):
    return labels_mapping.get(label)


def split(X, Y, test_size=0.3, random_state=42):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test


labels_mapping = {
    'dumbbell-chest-press': 'dumbbell-chest-press',
    'dumbbell-bench-press': 'dumbbell-chest-press',
    'biceps-curl': 'biceps-curl',
    'bicep curls': 'biceps-curl',
    'bicep': 'biceps-curl',
    'bice': 'biceps-curl',
    'rope-tricep-pushdown': 'rope-tricep-pushdown',
    'rope-tricep-pushdown ': 'rope-tricep-pushdown',
    'Push-up': 'push-up',
    'push-up': 'push-up',
    'triceps-extension': 'triceps-extension',
    'lateral-raise': 'lateral-raise',
    'lateral raises': 'lateral-raise',
}


labels = list(set(labels_mapping.values()))


all_labels = ['hiit-running-machine',
              'warm-up-cross-trainer',
              'dumbbell-chest-press',
              'dumbbell-shoulder-press',
              'barbell-biceps-curl',
              'rope-tricep-pushdown',
              'lateral-raise',
              'suitcase-crunches',
              'back-extensions',
              'suitcase-crunches ',
              'biceps-curl',
              'triceps-extension',
              'dumbbell-bench-press',
              'lateral-pulldown-straight',
              'cross trainer',
              'angle-chest-press',
              'seated row ',
              'seated row',
              'running-machine-hiit',
              'dumbbell-press',
              'triceps pull down ',
              'dumbbell shoulder press',
              'vertical swing',
              'vertical swing ',
              'dumbbell shoulder press ',
              'dumbbell-chest-fly',
              'running-machine-hit',
              'vertical-swing',
              'cable cross over',
              'run',
              'leg press',
              'leg press ',
              'crunch',
              'side-dips',
              'twist',
              'cable-crunch',
              'hiit',
              'side-dips/x',
              'pulldown-crunch',
              'twist/x',
              'leverage-high-row',
              'chest-fly',
              'chest-cable-cross-overs',
              'dumbbell-row',
              'cross-trainer-warm-up',
              'barbell-deadlift',
              'cable-straight-arm-pulldown',
              'barbell-bent-over-row',
              'cable-reverse-curls',
              'cable-reverse-curl',
              'sit',
              'walk',
              'stand',
              'front-lunge',
              'back-lunge',
              'kettlebell-swing-squat',
              'dumbbell-squat',
              'rope-tricep-pushdown ',
              'run ',
              'lat-pulldown-straight',
              'deltoid-row',
              'lat-pulldown-angled',
              'cable-cross-overs',
              'lateral raises',
              'bicep curls',
              'bicep',
              'lateral',
              'tc',
              'chest press',
              'shoulder press',
              'biceps curls (left)',
              'hiit running machine',
              'BC',
              'BC ',
              'test',
              'tesy',
              'bice',
              'first',
              'four',
              'second',
              'third',
              'LR ',
              'LR',
              'walking',
              't',
              'TE',
              'tc ',
              'oblique-crunches',
              'iron-cross',
              'squat',
              'triceps-dips',
              'dumbbell-calf-raise',
              'reverse-flyes',
              'Bench-press',
              'triceps-pushdown',
              'barbell-curl',
              'running-machine',
              'Dumbbell-flyes ',
              'reverse-triceps-Bench-press',
              'Chin-Up',
              'Reverse-Push-up',
              'Push-up',
              'push-up',
              'crunches',
              'bent-arm-barbell-pullover',
              'barbell-squat',
              'butterfly-machine',
              'bench-press',
              'barbell-row',
              'pull-up',
              'chest-press-machine',
              'overhead-cable-curl',
              'alternate-heel-touchers',
              'reverse-cable-curl',
              'upright-barbell-row',
              'cable-crossover',
              'lateral-pulldown']
