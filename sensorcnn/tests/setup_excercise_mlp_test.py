import sys
from functools import reduce
import numpy as np
import pandas as pd

from models.MLP import MLP
from dataset.examples import *


def test_mlp_balanced_data():
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
        set_setup_column(s, start=0, end=50)
        sys.stdout.write("%s " % i)

    print("Group all samples in one dataframe")
    all_samples_df = reduce(lambda x, y: x.append(y), all_samples.values())
    del all_samples
    print(all_samples_df['label'].value_counts())
    print(all_samples_df['setup_label'].value_counts())
    print("--------------------------------------------------------------")

    print("Setting numeric targets")
    setup_labels = ["setup_%s" % l for l in labels]
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
    model = MLP("150 id 64 Tanh_0 32 Tanh_1 6 Tanh_2")
    model.fit(X_train, Y_train, nb_epoch=30, batch_size=128)

    # Printing the results
    train_score = model.evaluate(X_train, Y_train, batch_size=128)
    test_score = model.evaluate(X_test, Y_test, batch_size=128)
    print(train_score)
    print(test_score)

    # Export the model
    model.save_weights("output/mlp/weights.raw")
    model.save_layers("output/mlp/layers.txt")
    with open("output/mlp/labels.txt", 'w') as f:
        f.writelines("\n".join(setup_labels))
