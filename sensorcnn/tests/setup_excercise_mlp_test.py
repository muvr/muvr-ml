import sys
from functools import reduce
import numpy as np
import pandas as pd

from models.MLP import MLP
from dataset.examples import *
from train.utils import *


def cache_get():
    try:
        x = np.load("/tmp/setup_exercise_mlp_test_data_x.npy")
        y = np.load("/tmp/setup_exercise_mlp_test_data_y.npy")
        print("Found Cached data")
        return x, y

    except:
        print("Cached data is missing")
        return None


def cache_put(data):
    x, y = data
    np.save("/tmp/setup_exercise_mlp_test_data_x", x)
    np.save("/tmp/setup_exercise_mlp_test_data_y", y)


def prepare_data():
    data = cache_get()
    if not data:
        print("Loading CSV files")
        all_samples = {}
        for filename in csv_file_iterator("/data"):
            samples = load_from_csv(filename)
            all_samples[filename] = samples

        print("Selecting the 6 exercises")
        for s in all_samples.values():
            s['label'] = s['label'].map(map_labels)

        print("Add setup exercise column")
        for i, s in enumerate(all_samples.values()):
            set_setup_column(s, start=0, end=100)
            sys.stdout.write("%s " % i)

        print("Group all samples in one dataframe")
        all_samples_df = reduce(lambda x, y: x.append(y), all_samples.values())
        del all_samples
        print(all_samples_df['label'].value_counts())
        print(all_samples_df['setup_label'].value_counts())
        print("--------------------------------------------------------------")

        print("Setting numeric targets")
        label_idx_dict = {l: i for i, l in enumerate(setup_labels)}
        label_to_idx = lambda x: label_idx_dict.get(x, 0)
        all_samples_df['target'] = all_samples_df['setup_label'].map(label_to_idx)
        print(all_samples_df['target'].value_counts())
        print("--------------------------------------------------------------")

        print("Preparing features and labels")
        # Extract features
        samples = all_samples_df[['x', 'y', 'z']].as_matrix()
        positive_targets = all_samples_df[all_samples_df['target'] > 0]
        negative_targets = all_samples_df[all_samples_df['target'] == 0]
        targets = np.append(positive_targets[['target']].as_matrix(),
                            negative_targets[['target']].as_matrix(),
                            axis=0)
        # Samples augmentation
        X, y = augment_examples(samples, targets)
        data = X, y
        cache_put(data)

    return data

def mlp_balanced_data(config, name):
    X, y = prepare_data()
    # Examples encoding
    Y = np.zeros((y.shape[0], 6))
    iy = np.array([[i, y] for (i, y) in enumerate(y) if y > 0])
    Y[iy[:, 0], iy[:, 1]] = 1

    # Balance the dataset
    n = int(len(y) / len(iy))
    partition = len(iy)
    new_X = boost(X, n, partition)
    new_Y = boost(Y, n, partition)

    X_train, X_test, Y_train, Y_test = split(new_X, new_Y)

    print("Training the MLP model")
    model = MLP(config=config, name=name)
    model.fit(X_train, Y_train, nb_epoch=30, batch_size=128)

    # Printing the results
    train_score = model.evaluate(X_train, Y_train, batch_size=128)
    test_score = model.evaluate(X_test, Y_test, batch_size=128)
    print(train_score)
    print(test_score)

    # Export result into csv file
    y_true = np.array([np.argmax(yi) + 1 if np.sum(yi) > 0 else 0 for yi in Y_test])
    y_pred = np.array([np.argmax(yi) + 1 if np.sum(yi) > 0 else 0 for yi in model.predict(X_test)])
    evaluation_tabel = evaluate(y_true, y_pred, ["no_exercise"] + setup_labels)

    # Continue testing on the test set and export the model
    model.fit(X_test, Y_test, nb_epoch=30, batch_size=128)
    export(model, evaluation_tabel, setup_labels)


def test_mlp_balanced_data_1():
    mlp_balanced_data("150 id 64 Tanh_0 32 Tanh_1 6 Tanh_2", "MLP1")


def test_mlp_balanced_data_2():
    mlp_balanced_data("150 id 100 Tanh_0 50 Tanh_1 6 Tanh_2", "MLP2")


def test_mlp_balanced_data_3():
    mlp_balanced_data("150 id 128 Tanh_0 46 Tanh_1 32 Tanh_2 6 Tanh_3", "MLP3")
